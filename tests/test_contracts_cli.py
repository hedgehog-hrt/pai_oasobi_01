import copy
import json

import pytest

from rosa_repro.cli import main
from rosa_repro.contracts import load_json, validate_manifest
from rosa_repro.domain import Problem
from helpers import fixture


def manifest():
    return {'schema_version':'1.0','run_id':'r','mode':'live_simulation',
            'measurement_phase':'formal','source_sha256':'a'*64,'git_commit':None,
            'git_dirty':None,'git_diff_sha256':None,'dependency_lock_sha256':'b'*64,
            'python_version':'3.12.14','clock_domain':'pod-boot-id','gpu_count':1,
            'gpus':[{'uuid':'GPU-1','kind':'test','memory_mib':96000,'driver':'test'}],
            'model_runtime':[{'component_id':'s1','revision':'test-revision','dtype':'bfloat16',
                              'torch_compile':False,'cuda_graphs':False,'attention_backend':'test',
                              'input_schema_sha256':'c'*64}],
            'initial_states_sha256':'d'*64,'observation_manifest_sha256':None,
            'profile_manifest_sha256':None,'unknown_fields':[]}


def test_valid_manifest():
    validate_manifest(manifest())


@pytest.mark.parametrize('case',['dtype','initial_states','dirty','gpu','unknown'])
def test_formal_run_does_not_hide_missing_evidence(case):
    d=manifest()
    if case=='dtype': d['model_runtime'][0]['dtype']=None
    if case=='initial_states': d['initial_states_sha256']=None
    if case=='dirty': d['git_dirty']=True
    if case=='gpu': d['gpu_count']=2
    if case=='unknown': d['unknown_fields']=['runtime.dtype']
    with pytest.raises(ValueError): validate_manifest(d)


def test_exploratory_unknowns_preserved():
    d=manifest(); d['measurement_phase']='exploratory'
    d['model_runtime'][0]['dtype']=None
    d['initial_states_sha256']=None
    d['unknown_fields']=['initial_states_sha256','model_runtime.s1.dtype']
    validate_manifest(d)


def test_measured_profile_requires_evidence():
    d=fixture(); p=d['profiles'][0]; p['source']='measured'
    p['points'][0]['rate_hz']=.1
    with pytest.raises(ValueError,match='evidence'): Problem.from_dict(d)
    p['evidence_sha256']='a'*64
    for point in p['points']: point.update(sample_count=1000,duration_s=120)
    Problem.from_dict(d)


def test_json_duplicate_keys_and_nonfinite_rejected(tmp_path):
    for content in ['{"a":1,"a":2}','{"a":NaN}']:
        p=tmp_path/'bad.json'; p.write_text(content)
        with pytest.raises(ValueError): load_json(p)


def test_cli_roundtrip_and_no_overwrite(tmp_path):
    p=tmp_path/'input.json'; p.write_text(json.dumps(fixture()))
    o=tmp_path/'out.json'
    assert main(['solve',str(p),'--output',str(o)])==0
    original=o.read_bytes()
    assert json.loads(original)['status']=='feasible'
    assert main(['solve',str(p),'--output',str(o)])==2
    assert o.read_bytes()==original


@pytest.mark.parametrize('pipeline',['P2','P3','P4'])
def test_pipeline_cli_and_refuse_overwrite(tmp_path,pipeline):
    from multi_helpers import fixture
    source=tmp_path/'input.json';source.write_text(json.dumps(fixture(pipeline)))
    output=tmp_path/'result.json'
    assert main(['evaluate-pipeline',str(source),'--output',str(output)])==0
    result=json.loads(output.read_text())
    assert result['qualified_action_chunks']==1 and len(result['input_sha256'])==64
    assert main(['evaluate-pipeline',str(source),'--output',str(output)])==2


def test_heterogeneous_cli_search_fixed_rates_and_oracle(tmp_path):
    from hetero_helpers import weighted_fixture
    source=tmp_path/'input.json';source.write_text(json.dumps(weighted_fixture()))
    rates=tmp_path/'rates.json';rates.write_text(json.dumps(dict(high=2.,low=1.)))
    grid=tmp_path/'grid.json';grid.write_text(json.dumps(dict(high=[1.,2.,3.],low=[1.,2.,3.])))
    for name,args in [('search',[]),('fixed',['--rates',str(rates)])]:
        output=tmp_path/f'{name}.json'
        assert main(['solve-heterogeneous',str(source),*args,'--output',str(output)])==0
        result=json.loads(output.read_text());assert result['status']=='feasible'
        assert result['gpu_execution'] is False and len(result['input_sha256'])==64
        if name=='fixed':assert result['weighted_objective']==7 and len(result['rates_sha256'])==64
    output=tmp_path/'oracle.json'
    assert main(['oracle-heterogeneous',str(source),'--grid',str(grid),'--output',str(output)])==0
    result=json.loads(output.read_text());assert result['status']=='grid_optimal'
    assert len(result['grid_sha256'])==64
    assert main(['oracle-heterogeneous',str(source),'--grid',str(grid),'--output',str(output)])==2
