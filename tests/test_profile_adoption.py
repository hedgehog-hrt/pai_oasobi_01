"""GPU metadata here is fabricated for contract tests only; no GPU result is saved."""
import json,shutil
from copy import deepcopy
import pytest
from test_profile_preparation import formal_fixture
from rosa_repro.profile_adoption import inspect,adopt,review_template,validate_adoption,inside,runtime_signature,validate_fleet_adoptions
from rosa_repro.profile_runner import seal
from rosa_repro.runtime_attestation import summarize_trace
from rosa_repro.profile_protocol import build_corpus
from rosa_repro.observation_quality import prepare_quality,seal_predictions,rubric_template
from rosa_repro.execution_plan import runtime_source_manifest,digest,compile_execution_plan
from rosa_repro.regression import write_json,save_observation
from rosa_repro.dummy import observation
from rosa_repro.paired import digest_value
from runtime_helpers import plan_for


def read(p):return json.loads(p.read_text())
def update(p,v):p.write_text(json.dumps(v))
def reseal(p):(p/'evidence.json').unlink();seal(p)


def fixture(tmp_path):
    paths=[formal_fixture(tmp_path,i) for i in range(3)];c=read(paths[0]/'condition.json')['config']
    images=tmp_path/'images'
    for i,split in enumerate(('profile','evaluation','holdout','holdout','holdout')):
        save_observation(images/str(i),observation(i+1),dict(task='FABRICATED TEST ONLY',case={'state_sha256':digest_value([i])},capture_split=split,execution_kind='fixture_metadata'))
    corpus=build_corpus(images)
    env=dict(python='fixture',python_binary_sha256='a'*64,system='Linux',machine='x86_64',packages={'torch':'test','ray':'test'},selected_environment={},
        tools={'nvidia_smi':{'status':'ok','stdout':'GPU-fake, Test GPU, test-only, 24000\n'}})
    backend=dict(model_artifacts_verified=True,attestation={'observed_torch_state':{'cuda_available':True,'devices':[{'name':'Test GPU','uuid':'GPU-fake','logical_index':0}],'matmul_allow_tf32':False}})
    for p in paths:
        x=read(p/'condition.json');x['corpus_id']=corpus['corpus_id'];update(p/'condition.json',x)
        r=read(p/'result.json');r['worker_environment']=env;r['worker_after']['backend_attestation']=backend;update(p/'result.json',r);reseal(p)
    k=tmp_path/'kernel';k.mkdir();write_json(k/'trace.json',{'traceEvents':[{'ph':'X','cat':'kernel','name':'FAKE_TEST_KERNEL','dur':1}]})
    write_json(k/'condition.json',dict(config=c,corpus_id=corpus['corpus_id'],split='evaluation',batch_size=1,
        observation_sha256=next(x['sha256'] for x in corpus['rows'] if x['split']=='evaluation')))
    write_json(k/'source_manifest.json',runtime_source_manifest());(k/'requests.jsonl').write_text('')
    write_json(k/'result.json',dict(status='complete',kind='gpu_backend_kernel_diagnostic',process_exit_code=0,process_timed_out=False,process_interrupted=False,
        gpu_execution=True,pretrained_weights_loaded=True,environment_after=env,backend=backend,traces={'trace.json':summarize_trace(k/'trace.json')}));seal(k)
    q=tmp_path/'quality';q.mkdir();s=prepare_quality(c,corpus,images);write_json(q/'condition.json',s);write_json(q/'source_manifest.json',runtime_source_manifest())
    rows=[dict(sample_id=x['sample_id'],status='ok',output={'plan':'fixture'}) for x in s['samples']]
    b=seal_predictions(dict(suite_id=s['suite_id'],mode='gpu',status='complete',drained=True,rows=rows,worker_environment=env,worker_after={'backend_attestation':backend}))
    write_json(q/'predictions.json',b);write_json(q/'result.json',{k:v for k,v in b.items() if k not in ('rows','bundle_id')});(q/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));seal(q)
    rubric=rubric_template(s,b)
    for r in rubric['rows']:r.update(reviewer='TEST ONLY',reason='Synthetic test assertion',checks={k:True for k in r['checks']})
    write_json(tmp_path/'ratings.json',rubric)
    return dict(schema_version='1.0',performance=[str(i) for i in range(3)],kernel='kernel',quality={'run':'quality','corpus_root':'images','ratings':'ratings.json'})


def approval(i):
    r=review_template(i);r['reviewer']='TEST ONLY'
    for c in r['checks'].values():c.update(accepted=True,reason='Synthetic fixture assertion; not a real approval')
    return r


def test_review_required_and_portable_accepted_identity(tmp_path):
    r=fixture(tmp_path);i=inspect(r,tmp_path)
    with pytest.raises(ValueError,match='reviewer'):adopt(r,tmp_path,review_template(i))
    a=adopt(r,tmp_path,approval(i));validate_adoption(a,tmp_path)
    moved=tmp_path/'copy';shutil.copytree(tmp_path,moved,ignore=shutil.ignore_patterns('copy'))
    assert validate_adoption(a,moved)==a
    a['profile']['points'][0]['p99_ms']=0
    with pytest.raises(ValueError,match='changed'):validate_adoption(a,tmp_path)


@pytest.mark.parametrize('bad',['cpu_trace','failed_process','kernel_config','kernel_environment','quality_missing_review','raw_quality','changed_source','missing_trial','changed_trace','stale_review'])
def test_adoption_rejects_missing_or_mismatched_evidence(tmp_path,bad):
    request=fixture(tmp_path);review=approval(inspect(request,tmp_path));k=tmp_path/'kernel';r=read(k/'result.json')
    if bad=='cpu_trace':
        update(k/'trace.json',{'traceEvents':[{'ph':'X','cat':'cpu_op','name':'x','dur':1}]});r['traces']={'trace.json':summarize_trace(k/'trace.json')};update(k/'result.json',r);reseal(k)
    elif bad=='failed_process':r['process_exit_code']=1;update(k/'result.json',r);reseal(k)
    elif bad=='kernel_config':x=read(k/'condition.json');x['config']['batch_size']=2;update(k/'condition.json',x);reseal(k)
    elif bad=='kernel_environment':r['environment_after']['packages']['torch']='different';update(k/'result.json',r);reseal(k)
    elif bad=='quality_missing_review':x=read(tmp_path/'ratings.json');x['rows'][0]['reviewer']=None;update(tmp_path/'ratings.json',x)
    elif bad=='raw_quality':(tmp_path/'quality/requests.jsonl').write_text('');reseal(tmp_path/'quality')
    elif bad=='changed_source':update(k/'source_manifest.json',{'sha256':'old'});reseal(k)
    elif bad=='missing_trial':request['performance']=request['performance'][:2]
    elif bad=='changed_trace':(k/'trace.json').write_text('{}')
    else:review['evidence_set_sha256']='0'*64
    with pytest.raises(ValueError):adopt(request,tmp_path,review)


@pytest.mark.parametrize('name',['../outside','/tmp/file'])
def test_escape_refused(tmp_path,name):
    with pytest.raises(ValueError):inside(tmp_path,name)


def test_symlink_evidence_refused(tmp_path):
    (tmp_path/'a').write_text('x');(tmp_path/'b').symlink_to(tmp_path/'a')
    with pytest.raises(ValueError):inside(tmp_path,'b')


def test_gpu_fleet_blocked_even_when_compiler_can_preview():
    cpu=plan_for(('P1',));x=cpu['inputs'];b={k:dict(factory='user_backend:factory',kwargs={}) for k in x['bindings']}
    gpu=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings=b)
    with pytest.raises(ValueError,match='adoptions'):validate_fleet_adoptions(gpu,None)
    assert validate_fleet_adoptions(cpu,None)=={}


def test_signature_preserves_flags_but_ignores_uuid_and_profiler_path(tmp_path):
    fixture(tmp_path);r=read(tmp_path/'kernel/result.json');env=r['environment_after'];b=r['backend'];s=runtime_signature(env,b)
    env['selected_environment']['VLLM_TORCH_PROFILER_DIR']='/new';b['attestation']['observed_torch_state']['devices'][0]['uuid']='other'
    assert runtime_signature(env,b)==s
    b['attestation']['observed_torch_state']['matmul_allow_tf32']=True
    assert runtime_signature(env,b)!=s

@pytest.mark.parametrize('package',['nvidia-cutlass-dsl','flashinfer-python','cuda-bindings','pyarrow'])
def test_signature_rejects_serving_binary_dependency_drift(tmp_path,package):
    fixture(tmp_path);r=read(tmp_path/'kernel/result.json');env=r['environment_after'];b=r['backend']
    env['packages'][package]='old';old=runtime_signature(env,b)
    env['packages'][package]='new';assert runtime_signature(env,b)!=old


def s1_quality_fixture(tmp_path,batch):
    from rosa_repro.profile_adoption import _quality,Evidence
    from rosa_repro.profile_protocol import template
    from rosa_repro.paired import save_state_bank,run_paired
    from rosa_repro.dummy import DummyEpisode,DummyTransport
    from rosa_repro.regression import sha256
    import numpy as np
    c=template('s1','groot');c['batch_size']=batch;bank=tmp_path/'bank'
    # Fabricated LIBERO/GPU metadata, local dummy execution, temporary contract fixture only.
    save_state_bank(bank,[np.array([1.,.2])],task='TEST',source={'kind':'LIBERO_benchmark_initial_states'})
    run_paired(bank=bank,output=tmp_path/'paired',env_factory=DummyEpisode,transport_factory=DummyTransport,execution_kind='libero_gpu')
    runtime=dict(observed_torch_state={'cuda_available':True,'devices':[{'name':'Test GPU'}]},observed_model_state={'parameter_tensor_counts_by_dtype':{'float32':1}})
    from array_helpers import Client,capture,inputs,StochasticPolicy
    server={'checkpoint':dict(revision=c['model_revision'],full_weight_hashes_verified=True),'runtime':runtime,'rosa_source_manifest':runtime_source_manifest()}
    client=Client(StochasticPolicy(),server)
    source=inputs(tmp_path/'inputs',2);assert capture(client,source,tmp_path/'reg')=='passed'
    from rosa_repro.profile_adoption import observed_signature
    signature=observed_signature(runtime)
    write_json(tmp_path/'association.json',dict(config_sha256=digest(c),corpus_id='fixture',source_manifest=runtime_source_manifest(),reviewer='TEST ONLY',reason='Fixture'))
    q={'quality':dict(regression='reg',paired='paired',state_bank='bank',association='association.json')}
    if batch>1:
        with pytest.raises(KeyError):_quality(Evidence(tmp_path),q,c,'fixture',signature)
        assert capture(client,source,tmp_path/'batch',batch_size=batch)=='passed';q['quality']['batch_regression']='batch'
        with pytest.raises(KeyError):_quality(Evidence(tmp_path),q,c,'fixture',signature)
        assert capture(client,source,tmp_path/'diagnostic',batch_size=batch,comparison='serial_vs_batch')=='mismatch';q['quality']['batch_diagnostic']='diagnostic'
    return q,c,signature


@pytest.mark.parametrize('batch',[1,2])
def test_s1_requires_matching_regression_and_raw_paired_outcomes(tmp_path,batch,monkeypatch):
    # This test isolates the earlier numerical/paired contract; R06 has its own raw-evidence tests.
    monkeypatch.setattr('rosa_repro.profile_adoption.r06_quality',lambda *args: {'test_scope':'numerical_contract_only'})
    from rosa_repro.profile_adoption import _quality,Evidence
    q,c,signature=s1_quality_fixture(tmp_path,batch)
    r=_quality(Evidence(tmp_path),q,c,'fixture',signature)
    assert r['pairs']==1 and not r['automatic_stochastic_batch_equivalence_claim']
    if batch>1:assert r['batch_diagnostic']['status']=='mismatch' and r['batch_regression']['numerical_admission_eligible']
    outcome=tmp_path/'paired/T01-0000/shared/outcome.json';x=read(outcome);x['success']=False;update(outcome,x)
    with pytest.raises(ValueError,match='summary differs'):_quality(Evidence(tmp_path),q,c,'fixture',signature)


@pytest.mark.parametrize('bad',['swapped_purpose','failed_diagnostic','changed_seed'])
def test_s1_diagnostic_is_required_complete_and_bound_to_regression(tmp_path,bad):
    from rosa_repro.profile_adoption import _quality,Evidence
    from rosa_repro.action_regression import seal_run
    q,c,signature=s1_quality_fixture(tmp_path,2)
    if bad=='swapped_purpose':q['quality']['batch_regression']='diagnostic'
    elif bad=='failed_diagnostic':
        p=tmp_path/'diagnostic/comparison_summary.json';v=read(p);v.update(status='failed',error={'type':'injected_failure'})
        update(p,v);(tmp_path/'diagnostic/regression_run.json').unlink();seal_run(tmp_path/'diagnostic')
    else:
        from array_helpers import Client,capture,StochasticPolicy
        capture(Client(StochasticPolicy(),read(tmp_path/'reg/server_evidence.json')),tmp_path/'inputs',tmp_path/'other',batch_size=2,comparison='serial_vs_batch',seed=43)
        q['quality']['batch_diagnostic']='other'
    with pytest.raises(ValueError):_quality(Evidence(tmp_path),q,c,'fixture',signature)


def test_gpu_binding_profile_and_stream_admission_checks(monkeypatch):
    cpu=plan_for(('P1',));x=cpu['inputs'];bindings={k:dict(factory='user_backend:factory',kwargs={}) for k in x['bindings']}
    plan=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings=bindings)
    w=plan['workers'][0];profile=next(p for p in x['fleet']['profiles'] if p['id']==w['profile_id'])
    c={k:w[k] for k in ('component_id','role','model_revision','runtime_fingerprint','hardware_kind','batch_size','batch_wait_ms')}
    c.update(binding=w['backend'],streams=len(w['members']),deadline_ms=x['fleet']['classes'][0]['components'][0]['deadline_ms'])
    a=dict(profile=profile,config=c)
    monkeypatch.setattr('rosa_repro.profile_adoption.read',lambda _:a)
    monkeypatch.setattr('rosa_repro.profile_adoption.validate_adoption',lambda value,root:value)
    b=dict(evidence_root='/tmp',adoptions={w['profile_id']:'fixture'})
    assert validate_fleet_adoptions(plan,b)
    a['config']['streams']=0
    with pytest.raises(ValueError,match='stream'):validate_fleet_adoptions(plan,b)
    a['config']['streams']=100;a['config']['binding']={'factory':'wrong:wrong','kwargs':{}}
    with pytest.raises(ValueError,match='binding'):validate_fleet_adoptions(plan,b)
