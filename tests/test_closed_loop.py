import asyncio,json,sys
from copy import deepcopy
import numpy as np
import pytest
from runtime_helpers import plan_for
from rosa_repro.closed_loop_protocol import protocol,prepare_suite,validate_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.local_fleet import LocalFleetTransport
from rosa_repro.simulator_process import ChangingDummyEpisode,ProcessSimulator
from rosa_repro.execution_plan import runtime_source_manifest,compile_execution_plan
from rosa_repro.paired import save_state_bank
from rosa_repro.regression import write_json


class Sim:
    def __init__(self):self.env=ChangingDummyEpisode();self.closed=False;self.starts=0
    async def start(self):return dict(kind='cpu_dummy',source_manifest=runtime_source_manifest())
    async def reset(self,state,seed):
        self.starts+=1;obs,h=self.env.reset(state,seed);return dict(observation=obs,restored_state_sha256=h,success=False)
    async def step(self,action,n):return self.env.step(action,n)
    async def close(self):self.closed=True


def suite(tmp_path,pipelines=('P1','P2','P3','P4'),**options):
    plan=plan_for(pipelines);bank=tmp_path/'bank'
    save_state_bank(bank,[np.array([1.,.3]),np.array([4.,.7]),np.array([8.,.4])],task='CPU',source={'kind':'synthetic'})
    p=protocol(plan,'cpu_smoke',request_timeout_s=.5,sim_timeout_s=5,drain_timeout_s=1,**options)
    return prepare_suite(plan,{c['audit']['id']:'bank' for c in plan['classes']},tmp_path,p,repetitions=2,setup_budget_s=20)


def run(s,tmp_path,*,sim_type=Sim,transport_type=LocalFleetTransport):
    sims={r['robot_id']:sim_type() for c in s['plan']['classes'] for r in c['robots']}
    async def go():
        t=transport_type(s['plan'])
        try:return await run_closed_loop(s,tmp_path,t,sims,tmp_path/'run')
        finally:await t.drain()
    return asyncio.run(go()),sims


def test_live_p1_p4_updates_observations_resets_sessions_and_reaudits(tmp_path):
    s=suite(tmp_path);r,sims=run(s,tmp_path)
    assert r['status']=='complete',r
    assert r['episodes_started']>4 and all(x.closed and x.starts>1 for x in sims.values())
    report=reaudit_closed(tmp_path/'run');assert report['actions_revalidated']>0 and not r['formal_protocol_met']
    outcomes=[json.loads(x) for x in (tmp_path/'run/episodes.jsonl').read_text().splitlines()]
    assert any(o['termination']=='success' and o['actual_steps']==11 for o in outcomes)
    assert all(o['actual_steps']<=720 and not o['steps_unknown'] for o in outcomes)
    requests=[json.loads(x) for x in (tmp_path/'run/requests.jsonl').read_text().splitlines()]
    byid={x['request_id']:x for x in requests}
    for x in requests:
        if x['plan_request_id']:assert byid[x['plan_request_id']]['episode_id']==x['episode_id']
    details=[json.loads(x) for x in (tmp_path/'run/request_details.jsonl').read_text().splitlines()]
    assert any(x['observation_version']>0 for x in details)
    assert len(list((tmp_path/'run').glob('*-action.npz')))==report['actions_revalidated']


@pytest.mark.parametrize('bad',['steps','observation','reset','timeout'])
def test_simulator_failure_retains_partial_evidence_and_stops(tmp_path,bad):
    class Bad(Sim):
        async def reset(self,state,seed):
            if bad=='reset':raise RuntimeError('fixture reset failure')
            return await super().reset(state,seed)
        async def step(self,action,n):
            if bad=='timeout':raise TimeoutError('fixture simulator timeout')
            r=await super().step(action,n)
            if bad=='steps':r['steps']=n+1
            if bad=='observation':r['observation']={}
            return r
    s=suite(tmp_path,('P1',));r,sims=run(s,tmp_path,sim_type=Bad)
    assert r['status']=='failed' and all(x.closed for x in sims.values())
    assert (tmp_path/'run/evidence.json').exists()
    if bad in ('steps','timeout'):
        o=json.loads((tmp_path/'run/episodes.jsonl').read_text().splitlines()[0]);assert o['steps_unknown']


@pytest.mark.parametrize('limit', ['request','episode','capture'])
def test_caps_never_turn_early_stop_into_success(tmp_path,limit):
    kwargs={'max_requests':1} if limit=='request' else {'max_episodes_per_robot':1} if limit=='episode' else {'max_capture_bytes':1}
    s=suite(tmp_path,('P1',),**kwargs);r,_=run(s,tmp_path);assert r['status']=='failed' and not r.get('formal_protocol_met',False)


def test_model_timeout_no_late_action_and_all_simulators_close(tmp_path):
    class Slow(LocalFleetTransport):
        async def predict(self,*args):await asyncio.sleep(.1);return await super().predict(*args)
    s=suite(tmp_path,('P1',));p=protocol(s['plan'],'cpu_smoke',request_timeout_s=.01,sim_timeout_s=1,drain_timeout_s=.5)
    s=prepare_suite(s['plan'],{'c0':'bank'},tmp_path,p,repetitions=1,setup_budget_s=20)
    r,_=run(s,tmp_path,transport_type=Slow)
    assert r['status']=='failed' and not list((tmp_path/'run').glob('*-action.npz'))
    assert any(json.loads(x)['status']=='timeout' for x in (tmp_path/'run/requests.jsonl').read_text().splitlines())


def test_fixed_window_sample_shortfall_is_explicit(tmp_path):
    s=suite(tmp_path,('P2',),duration_s=.001,warmup_s=.001)
    r,_=run(s,tmp_path);assert r['status']=='measurement_shortfall' and not r['sample_requirements_met']


def test_suite_is_state_and_source_bound(tmp_path):
    s=suite(tmp_path);validate_suite(s,tmp_path)
    s['repetitions']=9
    with pytest.raises(ValueError):validate_suite(s,tmp_path)


def test_gpu_formal_duration_and_repetition_guards(tmp_path):
    cpu=plan_for(('P4',));x=cpu['inputs'];plan=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings={k:{'factory':'test:backend','kwargs':{}} for k in x['roles']})
    p=protocol(plan);assert p['warmup_s']>=30 and p['duration_s']>=120 and p['min_measure_success_per_active_class_component']==1000
    with pytest.raises(ValueError):protocol(plan,warmup_s=1)
    with pytest.raises(ValueError):protocol(cpu,'formal')
    with pytest.raises(ValueError):prepare_suite(plan,{},tmp_path,p,repetitions=1)
    with pytest.raises(ValueError):protocol(plan,warmup_min_success=19)


def test_actual_simulator_process_reset_step_and_close(tmp_path):
    async def go():
        sim=ProcessSimulator({'kind':'cpu_dummy'},python=sys.executable,stderr_path=tmp_path/'sim.log',timeout_s=5)
        try:
            r=await sim.start();assert r['kind']=='cpu_dummy'
            reset=await sim.reset(np.array([2.]),42)
            from rosa_repro.dummy import DummyPolicy
            action=DummyPolicy().get_action(reset['observation'])[0]
            result=await sim.step(action,8);assert result['steps']==8 and result['observation']['state.x'][0,0,0]==10
        finally:await sim.close()
        assert sim.proc.poll() is not None and sim.closed
    asyncio.run(go())


@pytest.mark.parametrize('bad',['telemetry','cleanup'])
def test_cleanup_or_telemetry_errors_cannot_pass(tmp_path,bad):
    class BadSim(Sim):
        async def close(self):
            await super().close()
            if bad=='cleanup':raise RuntimeError('fixture cleanup error')
    class BadTransport(LocalFleetTransport):
        async def telemetry(self,wid):
            if bad=='telemetry':raise RuntimeError('fixture telemetry error')
            return {'kind':'cpu_inprocess_fixture'}
    r,_=run(suite(tmp_path,('P1',)),tmp_path,sim_type=BadSim,transport_type=BadTransport)
    assert r['status']=='failed' and not r['formal_protocol_met']


@pytest.mark.parametrize('bad',['protocol','summary','steps','input'])
def test_closed_reaudit_rejects_resealed_inconsistent_evidence(tmp_path,bad):
    from rosa_repro.profile_runner import seal
    run(suite(tmp_path,('P1',)),tmp_path);out=tmp_path/'run'
    if bad=='protocol':
        p=out/'protocol.json';x=json.loads(p.read_text());x['duration_s']=100;p.write_text(json.dumps(x))
    elif bad=='summary':
        p=out/'result.json';x=json.loads(p.read_text());x['weighted_Q']+=1;p.write_text(json.dumps(x))
    elif bad=='steps':
        p=out/'episodes.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['actual_steps']+=1
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    else:
        p=out/'request_details.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['input_sha256']='0'*64
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (out/'evidence.json').unlink();seal(out)
    with pytest.raises(ValueError):reaudit_closed(out)


def test_live_reports_retain_outcomes_and_missing_repetition(tmp_path):
    from rosa_repro.comparison_report import condition,report
    run(suite(tmp_path,('P1',)),tmp_path)
    m=dict(schema_version='1.0',purpose='prospective_pipeline_comparison',conditions={'live':condition(tmp_path/'run')},
        cells=[dict(id='r'+str(i),condition_id='live',variant='r07',repetition=i,directory=d) for i,d in enumerate(('run','missing'))])
    r=report(m,tmp_path)
    assert [x['status'] for x in r['rows']]==['passed','missing']
    assert r['rows'][0]['closed_loop_audit']['tasks']['classes']['c0']['measurement_started']>0
    assert not r['paper_reproduction_confirmed']


def test_warmup_shortfall_stops_before_measurement_and_reaudits(tmp_path):
    import time
    from rosa_repro.comparison_report import audit_pipeline
    s=suite(tmp_path,('P1','P4'),warmup_s=.08,duration_s=10,warmup_min_success=100)
    started=time.monotonic();result,sims=run(s,tmp_path);elapsed=time.monotonic()-started
    assert result['status']=='warmup_shortfall' and not result['measurement_started'] and result['drained'],result
    assert result['weighted_Q'] is None and result['qualified_action_chunks'] is None and not result['formal_protocol_met']
    assert elapsed<3 and all(sim.closed for sim in sims.values())
    checked=reaudit_closed(tmp_path/'run');gate=checked['warmup_gate'];assert gate['shortfall'] and not gate['requirements_met']
    records=json.loads((tmp_path/'run/audit.json').read_text())['records']
    assert all((r['sent_ns'] if r['record_type']=='request' else r['start_ns'])<gate['cutoff_ns'] for r in records)
    assert audit_pipeline(tmp_path/'run')['status']=='failed'


def test_successful_warmup_admits_fixed_window_without_resetting_epoch(tmp_path):
    s=suite(tmp_path,('P1','P4'),warmup_s=.2,duration_s=.3,warmup_min_success=1)
    r,_=run(s,tmp_path);assert r['status']=='complete' and r['measurement_started'],r
    a=reaudit_closed(tmp_path/'run');assert a['warmup_gate']['requirements_met'] and a['metrics']['duration_s']==.3
    assert all(n>=1 for n in a['warmup_gate']['counts'].values())


def test_late_warmup_model_result_is_drained_and_never_applied(tmp_path):
    class Late(LocalFleetTransport):
        async def predict(self,*args):await asyncio.sleep(.15);return await super().predict(*args)
    s=suite(tmp_path,('P1',),warmup_s=.02,duration_s=5,warmup_min_success=1)
    r,_=run(s,tmp_path,transport_type=Late)
    assert r['status']=='warmup_shortfall' and r['drained'] and not list((tmp_path/'run').glob('*-action.npz'))
    checked=reaudit_closed(tmp_path/'run');assert checked['warmup_gate']['counts']=={'c0/s1':0}


def test_inflight_simulator_stop_retains_unknown_steps_and_finalizes_episode(tmp_path):
    class SlowSim(Sim):
        async def step(self,action,n):await asyncio.sleep(.3);return await super().step(action,n)
    s=suite(tmp_path,('P1',),warmup_s=.05,duration_s=5,warmup_min_success=100)
    r,sims=run(s,tmp_path,sim_type=SlowSim)
    assert r['status']=='warmup_shortfall' and all(x.closed for x in sims.values())
    rows=[json.loads(x) for x in (tmp_path/'run/episodes.jsonl').read_text().splitlines()]
    assert rows and rows[0]['steps_unknown'] and rows[0]['end_ns']>rows[0]['start_ns']
    assert reaudit_closed(tmp_path/'run')['actions_revalidated']==0


def test_cleanup_failure_is_preserved_when_warmup_is_rejected(tmp_path):
    class BadSim(Sim):
        async def close(self):raise RuntimeError('injected cleanup failure')
    s=suite(tmp_path,('P1',),warmup_s=.05,duration_s=5,warmup_min_success=100)
    r,_=run(s,tmp_path,sim_type=BadSim)
    assert r['status']=='failed' and not r['measurement_started'] and r['simulator_cleanup_errors']
    assert not reaudit_closed(tmp_path/'run')['warmup_gate']['requirements_met']


def test_warmup_counts_exclude_errors_late_completions_and_boundary_requests(tmp_path):
    from rosa_repro.closed_loop_protocol import warmup_check
    s=suite(tmp_path,('P1',),warmup_min_success=2);rid=s['plan']['classes'][0]['robots'][0]['robot_id']
    base=dict(record_type='request',robot_id=rid,component_id='s1',status='ok',valid_output=True,sent_ns=1,finished_ns=9)
    rows=[base,dict(base,status='error',valid_output=False),dict(base,finished_ns=10),dict(base,sent_ns=10,finished_ns=11)]
    r=warmup_check(s['protocol'],s['plan']['classes'],rows,10)
    assert r['counts']=={'c0/s1':1} and not r['requirements_met']


def test_each_active_component_must_pass_warmup(tmp_path):
    from rosa_repro.closed_loop_protocol import warmup_check
    s=suite(tmp_path,('P4',),warmup_min_success=1);rid=s['plan']['classes'][0]['robots'][0]['robot_id']
    rows=[dict(record_type='request',robot_id=rid,component_id=m['id'],status='ok',valid_output=True,sent_ns=1,finished_ns=2)
        for m in s['plan']['classes'][0]['audit']['components'] if m['role']!='monitor']
    r=warmup_check(s['protocol'],s['plan']['classes'],rows,3)
    assert r['shortfall']==['c0/monitor']


@pytest.mark.parametrize('bad',['counts','result','new_request'])
def test_resealed_warmup_admission_forgery_rejected(tmp_path,bad):
    from rosa_repro.profile_runner import seal
    s=suite(tmp_path,('P1',),warmup_s=.08,duration_s=5,warmup_min_success=100);run(s,tmp_path);out=tmp_path/'run'
    if bad=='counts':
        p=out/'warmup_gate.json';x=json.loads(p.read_text());x['counts']['c0/s1']+=1;p.write_text(json.dumps(x))
    elif bad=='result':
        p=out/'result.json';x=json.loads(p.read_text());x.update(status='complete',measurement_started=True);p.write_text(json.dumps(x))
    else:
        p=out/'audit.json';x=json.loads(p.read_text());x['records'][0]['sent_ns']=x['window_start_ns'];p.write_text(json.dumps(x))
    (out/'evidence.json').unlink();seal(out)
    with pytest.raises(ValueError):reaudit_closed(out)
