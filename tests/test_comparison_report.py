import json,shutil
from copy import deepcopy
import pytest
from test_pipeline_runtime import run,bindings
from runtime_helpers import plan_for
from rosa_repro.comparison_report import condition,report,markdown,stats,validate_manifest


def manifest(root,dirs=('rep0','rep1')):
    return dict(schema_version='1.0',purpose='prospective_pipeline_comparison',conditions={'cpu':condition(root/dirs[0])},
        cells=[dict(id=x,condition_id='cpu',variant='r07',repetition=i,directory=x) for i,x in enumerate(dirs)])


def test_recomputed_p1_p4_tables_and_repetition_statistics(tmp_path):
    p=plan_for(('P1','P2','P3','P4'))
    for name in ('rep0','rep1'):run(p,tmp_path/name)
    r=report(manifest(tmp_path),tmp_path);assert r['status']=='complete' and not r['paper_reproduction_confirmed']
    assert {x['pipeline'] for x in r['groups'][0]['classes'].values()}=={'P1','P2','P3','P4'}
    assert r['groups'][0]['observed_weighted_Q']['n']==2 and r['groups'][0]['observed_weighted_Q']['sample_stddev'] is not None
    assert 'Predicted objective' in markdown(r)


def test_missing_failure_and_invalid_condition_retained(tmp_path):
    run(plan_for(),tmp_path/'ok');run(plan_for(bindings=bindings(fail_calls=[1])),tmp_path/'failed')
    m=manifest(tmp_path,('ok','failed','missing'));r=report(m,tmp_path)
    assert [x['status'] for x in r['rows']]==['passed','failed','missing']
    assert r['status']=='incomplete' and not r['groups'][0]['complete'] and r['groups'][0]['observed_weighted_Q']['n']==1
    m['conditions']['cpu']['execution_mode']='gpu'
    assert report(m,tmp_path)['rows'][0]['status']=='invalid'


@pytest.mark.parametrize('kind',['raw','summary','source','duplicate_run','duplicate_cell'])
def test_corruption_and_duplicate_evidence_cannot_enter_average(tmp_path,kind):
    run(plan_for(('P1',)),tmp_path/'rep0');m=manifest(tmp_path,('rep0',))
    if kind=='raw':(tmp_path/'rep0/events.jsonl').write_text('')
    elif kind=='summary':(tmp_path/'rep0/metrics.json').write_text('{}')
    elif kind=='source':(tmp_path/'rep0/source_manifest.json').write_text('{}')
    elif kind=='duplicate_run':
        shutil.copytree(tmp_path/'rep0',tmp_path/'copy');m['cells'].append(dict(m['cells'][0],id='copy',repetition=1,directory='copy'))
    else:
        m['cells'].append(deepcopy(m['cells'][0]))
        with pytest.raises(ValueError):validate_manifest(m)
        return
    r=report(m,tmp_path);assert r['status']=='incomplete' and r['rows'][-1]['status']=='invalid'


def test_statistics_are_between_repetitions_and_singleton_uncertain():
    assert stats([2,4])['mean']==3 and stats([2,4])['sample_stddev']==pytest.approx(2**.5)
    assert stats([2])['sample_stddev'] is None and stats([])['mean'] is None
