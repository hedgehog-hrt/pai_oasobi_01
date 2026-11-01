import json,sys
from pathlib import Path
import pytest
from test_profile_preparation import config,corpus
from rosa_repro.profile_protocol import make_suite
from rosa_repro.regression import write_json
from rosa_repro.resume_workflow import prepare,check,run,supervise
from rosa_repro.execution_plan import runtime_source_manifest


def plan(tmp_path):
    root,book=corpus(tmp_path);c=config();write_json(tmp_path/'config.json',c);suites={}
    for stage in ('smoke','pilot','profile'):
        p=tmp_path/(stage+'.json');write_json(p,make_suite(c,book,stage=stage,phase='cpu_smoke',rates=[10] if stage=='profile' else [],repetitions=2 if stage=='profile' else 1));suites[stage]=p
    return prepare(tmp_path/'config.json',root,suites,driver_python=sys.executable,worker_python=sys.executable,setup_budget_s=10,kernel_budget_s=10)


def test_default_check_has_no_execution_or_output_dir(tmp_path,monkeypatch):
    p=plan(tmp_path)
    monkeypatch.setattr('rosa_repro.resume_workflow.supervise',lambda *a:pytest.fail('No processes expected'))
    r=run(p,tmp_path/'run');assert r['status']=='planned' and not (tmp_path/'run').exists()
    assert [s['id'] for s in p['steps']]==['environment','model-preflight','smoke-0','kernel','pilot-0','profile-0','profile-1']


def test_budget_prevents_any_step_and_invalid_budgets_refused(tmp_path):
    p=plan(tmp_path);r=run(p,tmp_path/'run',execute=True,budget_s=1)
    assert r['status']=='budget_stop' and r['executed_steps']==0
    for value in (None,0,-1,float('nan')):
        with pytest.raises(ValueError):run(p,tmp_path/'run',execute=True,budget_s=value)


def test_failed_step_retried_without_repeating_success_and_tamper_refused(tmp_path,monkeypatch):
    p=plan(tmp_path);calls=[]
    def process(argv,env,out,timeout):
        calls.append(out.parent.name)
        if out.parent.name=='environment':write_json(out/'environment.json',{'source_manifest':runtime_source_manifest()})
        return dict(exit_code=0 if out.parent.name=='environment' else 2,timed_out=False,interrupted=False)
    monkeypatch.setattr('rosa_repro.resume_workflow.supervise',process)
    r=run(p,tmp_path/'run',execute=True,budget_s=1000,max_new=3);assert r['status']=='failed' and calls==['environment','model-preflight']
    r=run(p,tmp_path/'run',execute=True,budget_s=1000,max_new=3);assert calls==['environment','model-preflight','model-preflight']
    assert (tmp_path/'run/model-preflight/attempt-002/result.json').exists()
    (tmp_path/'run/environment/attempt-001/environment.json').write_text('{}')
    with pytest.raises(ValueError,match='modified'):run(p,tmp_path/'run',execute=True,budget_s=1000)


@pytest.mark.parametrize('kind',['config','corpus','python','plan'])
def test_changes_invalidate_resume_plan(tmp_path,kind):
    p=plan(tmp_path)
    if kind=='config':Path(p['config']).write_text('{}')
    elif kind=='corpus':next(Path(p['corpus_root']).rglob('arrays.npz')).write_bytes(b'bad')
    elif kind=='python':p['python_binary_hashes'][sys.executable]='bad'
    else:p['steps'][0]['timeout_s']=0
    with pytest.raises((ValueError,KeyError,Exception)):check(p)


def test_actual_subprocess_failure_and_timeout_are_preserved(tmp_path):
    a=tmp_path/'failure';a.mkdir()
    r=supervise([sys.executable,'-c','raise SystemExit(3)'],{},a,1);assert r['exit_code']==3 and not r['timed_out']
    b=tmp_path/'timeout';b.mkdir()
    r=supervise([sys.executable,'-c','import time;time.sleep(20)'],{},b,.05);assert r['timed_out'] and r['exit_code']!=0


def test_crashed_gpu_attempt_requires_evidence_bound_cleanup_review(tmp_path,monkeypatch):
    from rosa_repro.experiments import ExperimentStore
    p=plan(tmp_path);p['mode']='gpu';out=tmp_path/'run'
    monkeypatch.setattr('rosa_repro.resume_workflow.check',lambda _:dict(status='ready'))
    with ExperimentStore(out,dict(plan=p,ray_address='fixture-cluster')) as store:
        with pytest.raises(KeyboardInterrupt):
            with store.attempt('environment') as d:raise KeyboardInterrupt()
    r=run(p,out,execute=True,budget_s=1000,address='fixture-cluster')
    assert r['status']=='blocked_cluster_cleanup' and r['executed_steps']==0
    def process(argv,env,out,timeout):
        write_json(out/'environment.json',{'source_manifest':runtime_source_manifest()})
        return dict(exit_code=0,timed_out=False,interrupted=False)
    monkeypatch.setattr('rosa_repro.resume_workflow.supervise',process)
    review=dict(attempt_sha256=r['attempt_sha256'],cleanup_confirmed=True,reviewer='TEST ONLY',reason='No actual cluster exists')
    r=run(p,out,execute=True,budget_s=1000,address='fixture-cluster',cleanup_review=review)
    assert r['executed_steps']==1 and r['status']=='step_limit'
