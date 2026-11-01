import asyncio
from copy import deepcopy
import json
from pathlib import Path
import pytest
from rosa_repro.adapter import ProtocolError
from rosa_repro.component_runtime import ComponentCore,s1_observation
from rosa_repro.dummy import observation
from rosa_repro.execution_plan import DUMMY_FACTORY
from rosa_repro.local_fleet import LocalFleetTransport
from rosa_repro.pipeline_runtime import run_pipeline,reaudit
from runtime_helpers import plan_for


def load(path):return json.loads(path.read_text())

def bindings(**kwargs):return {c:dict(factory=DUMMY_FACTORY,kwargs=kwargs if c=='s1' else {}) for c in ('s1','s2','safety','monitor')}


def run(p,path,**kwargs):
    async def go():
        t=LocalFleetTransport(p)
        try:return await run_pipeline(p,t,lambda robot,seq:observation(1),path,warmup_s=.06,duration_s=.24,request_timeout_s=.3,**kwargs)
        finally:await t.drain()
    return asyncio.run(go())


@pytest.mark.parametrize('pipeline',['P2','P3','P4'])
def test_runtime_events_feed_evaluator_and_can_be_recomputed(tmp_path,pipeline):
    p=plan_for((pipeline,));out=tmp_path/'run';r=run(p,out)
    assert r['status']=='complete' and r['qualified_action_chunks']>0
    metrics=load(out/'metrics.json');again=reaudit(out,tmp_path/'recomputed.json');assert again==metrics
    audit=load(out/'audit.json');requests=[r for r in audit['records'] if r['record_type']=='request']
    assert {r['component_id'] for r in requests}=={m['id'] for m in p['classes'][0]['audit']['components']}
    goal=[r for r in requests if r['component_id']=='s1']
    assert all(b['sent_ns']-a['sent_ns']>=1e9/30 for a,b in zip(goal,goal[1:]))
    if pipeline=='P4':
        assert len([r for r in requests if r['component_id']=='s2'])>=2
        assert all(r['plan_request_id'] for r in goal)
        assert all(a['plan_use_index']<=3 for a in metrics['actions'])
        workers=load(out/'workers_after.json')
        assert sum(w['backend_plan_inputs'] or 0 for w in workers.values())==len(goal)
    with pytest.raises(FileExistsError):run(p,out)


def test_s2_text_injected_without_mutating_original_observation():
    obs=observation(3);p=dict(observation=obs,plan='open gripper')
    merged=s1_observation(p)
    assert merged['task']==['CPU fixture\nPlan: open gripper'] and obs['task']==['CPU fixture']


def test_failures_logged_and_no_action_applied_for_invalid_output(tmp_path):
    p=plan_for(bindings=bindings(fail_calls=[1],invalid_calls=[3]));out=tmp_path/'run'
    r=run(p,out);assert r['status']=='complete_with_request_failures' and r['request_statuses']['error']==2
    events=load(out/'audit.json')['records'];failed={e['request_id'] for e in events if e['record_type']=='request' and not e['valid_output']}
    assert not any(e['record_type']=='action' and e['request_id'] in failed for e in events)


def test_timeout_is_finalized_and_late_response_cannot_be_applied(tmp_path):
    p=plan_for(bindings=bindings(delay_s=.03))
    async def go():
        t=LocalFleetTransport(p)
        try:return await run_pipeline(p,t,lambda r,s:observation(1),tmp_path/'run',warmup_s=.06,duration_s=.15,request_timeout_s=.01)
        finally:await t.drain()
    # Drain timeout is intentionally shorter than late backend work: failure is honest.
    try:r=asyncio.run(go())
    except TimeoutError:r=load(tmp_path/'run/result.json')
    events=[json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    late={e['request_id'] for e in events if e['record_type']=='request' and e['status']=='timeout'}
    assert late and not any(e['record_type']=='action' and e['request_id'] in late for e in events)


def test_safety_stop_blocks_executor_but_preserves_decision_log(tmp_path):
    b=bindings();b['safety']['kwargs']={'stop_calls':list(range(1,100))}
    p=plan_for(bindings=b);r=run(p,tmp_path/'run')
    assert r['qualified_action_chunks']==0
    metrics=load(tmp_path/'run/metrics.json')
    assert metrics['classes']['c0']['safety_stopped_decisions']>0
    assert not any(a['applied'] for a in metrics['actions'])


def test_slow_periodic_worker_skips_slots_without_catchup_burst(tmp_path):
    b=bindings();b['safety']['kwargs']={'delay_s':.055};p=plan_for(bindings=b)
    run(p,tmp_path/'run')
    events=load(tmp_path/'run/audit.json')['records'];safety=[e for e in events if e['component_id']=='safety']
    assert any(b['slot_index']-a['slot_index']>1 for a,b in zip(safety,safety[1:]))
    metrics=load(tmp_path/'run/metrics.json')
    assert metrics['periodic_slot_metrics']['c0']['safety']['status_counts'].get('missing',0)>0
    assert any(s['status']=='violated' for s in metrics['slots'] if s['component_id']=='safety')


def test_wrong_response_identity_aborts_without_action_or_final_audit(tmp_path):
    p=plan_for(('P1',))
    class Wrong(LocalFleetTransport):
        async def predict(self,*args):
            r=await super().predict(*args);r['metadata']['robot_id']='wrong';return r
    async def go():
        t=Wrong(p)
        try:await run_pipeline(p,t,lambda r,s:observation(1),tmp_path/'run',warmup_s=0,duration_s=.1)
        finally:await t.drain()
    with pytest.raises(ProtocolError,match='mismatch'):asyncio.run(go())
    assert load(tmp_path/'run/result.json')['status']=='failed'
    assert not (tmp_path/'run/audit.json').exists()
    assert all(json.loads(line)['record_type']=='request' for line in (tmp_path/'run/events.jsonl').read_text().splitlines())


def test_executor_failure_keeps_intent_and_never_fabricates_action(tmp_path):
    async def fail(*args):raise RuntimeError('partial action unknown')
    p=plan_for(('P1',))
    with pytest.raises(RuntimeError,match='partial'):run(p,tmp_path/'run',execute=fail,action_semantics='external_executor_reported_steps')
    life=[json.loads(x) for x in (tmp_path/'run/lifecycle.jsonl').read_text().splitlines()]
    assert any(x['kind']=='action_started' for x in life) and life[-1]['kind']=='run_failed'
    with pytest.raises(FileNotFoundError):reaudit(tmp_path/'run',tmp_path/'metrics.json')


def test_reset_during_inference_cancels_old_session_but_other_robot_remains_live():
    p=plan_for(('P1','P1'));w=p['workers'][0];w=deepcopy(w);w['backend']['kwargs']={'delay_s':.03}
    c=ComponentCore(w,p['plan_id']);rid=w['members'][0]
    def metadata(rid,ack,q,seq=0):return dict(plan_id=p['plan_id'],worker_id=w['worker_id'],run_id='r',robot_id=rid,
        episode_id=ack['episode_id'],request_id=q,attempt_id=q,component_id='s1',observation_seq=seq,generation=ack['generation'])
    async def go():
        ack=c.reset('r',rid,'e0');meta=metadata(rid,ack,'q0');ticket=c.accept(meta,dict(observation=observation(1)))
        task=asyncio.create_task(c.run_batch([ticket]));await asyncio.sleep(.005)
        c.reset('r',rid,'e1');out=(await task)[0];c.finish(ticket)
        assert out['status']=='cancelled' and out['output'] is None
        other=w['members'][1];ack=c.reset('r',other,'e0')
        meta=metadata(other,ack,'q1');ticket=c.accept(meta,dict(observation=observation(2)))
        assert (await c.run_batch([ticket]))[0]['status']=='ok';c.finish(ticket)
        with pytest.raises(ProtocolError,match='Duplicate'):c.accept(meta,dict(observation=observation(2)))
    asyncio.run(go())


def test_batch_outputs_keep_robot_values_and_plan_inputs_separate():
    p=plan_for(('P1','P1'));w=deepcopy(p['workers'][0]);w['batch_size']=2
    c=ComponentCore(w,p['plan_id']);tickets=[]
    for i,rid in enumerate(w['members']):
        ack=c.reset('run',rid,'e')
        meta=dict(plan_id=p['plan_id'],worker_id=w['worker_id'],run_id='run',robot_id=rid,episode_id='e',
                  request_id=f'q{i}',attempt_id=f'a{i}',component_id='s1',observation_seq=0,generation=ack['generation'])
        tickets.append(c.accept(meta,dict(observation=observation(i+1),plan='move' if i==0 else None)))
    outputs=asyncio.run(c.run_batch(tickets))
    assert [float(r['output']['action.x'][0,0,0]) for r in outputs]==[1.,2.]
    assert len({r['batch_id'] for r in outputs})==1 and all(r['actual_batch_size']==2 for r in outputs)
    assert c.backend.plan_inputs==1
    for t in tickets:c.finish(t)


def test_s2_failure_never_sends_unplanned_s1_and_retries_with_new_id(tmp_path):
    b=bindings();b['s2']['kwargs']={'fail_calls':[1]};p=plan_for(bindings=b)
    r=run(p,tmp_path/'run');assert r['request_statuses']['error']==1
    events=load(tmp_path/'run/audit.json')['records']
    req=[e for e in events if e['record_type']=='request']
    plans={e['request_id'] for e in req if e['component_id']=='s2' and e['valid_output']}
    assert plans and all(e['plan_request_id'] in plans for e in req if e['component_id']=='s1')


def test_raw_event_tampering_and_request_budget_failure_are_not_completed_runs(tmp_path):
    p=plan_for(('P1',));run(p,tmp_path/'good')
    with (tmp_path/'good/events.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='disagree'):reaudit(tmp_path/'good',tmp_path/'badmetrics.json')
    with pytest.raises(RuntimeError,match='cap'):run(p,tmp_path/'limited',max_requests=1)
    assert load(tmp_path/'limited/result.json')['status']=='failed'
    assert not (tmp_path/'limited/metrics.json').exists()


def test_worker_restart_is_detected_at_end_even_without_request_failure(tmp_path):
    p=plan_for(('P1',))
    class Restarted(LocalFleetTransport):
        calls=0
        async def inventory(self):
            self.calls+=1;result=await super().inventory()
            if self.calls>1:
                for v in result.values():v['instance_id']='new-instance'
            return result
    async def go():
        t=Restarted(p)
        try:await run_pipeline(p,t,lambda r,s:observation(1),tmp_path/'run',duration_s=.08,warmup_s=0)
        finally:await t.drain()
    with pytest.raises(ProtocolError,match='restarted'):asyncio.run(go())
    assert load(tmp_path/'run/result.json')['status']=='failed'


def test_zero_goal_rate_still_collects_obligations_for_entire_window(tmp_path):
    from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler
    from rosa_repro.execution_plan import compile_execution_plan
    p=plan_for(('P2',));x=p['inputs']
    schedule=HeterogeneousScheduler(Fleet.from_dict(x['fleet'])).check_rates({'c0':0.})
    p=compile_execution_plan(x['fleet'],schedule,x['roles'])
    r=run(p,tmp_path/'run');assert r['qualified_action_chunks']==0
    a=load(tmp_path/'run/audit.json')
    assert all(e['component_id']=='monitor' for e in a['records'])
    assert a['episodes'][0]['end_ns']>=a['window_end_ns']
    assert len(a['records'])>=5


def test_structured_monitor_evidence_and_execution_history_reach_saved_logs(tmp_path):
    b=bindings();b['monitor']['kwargs']={'async_concurrency':2,'delay_s':.001}
    p=plan_for(bindings=b)
    captured=[]
    class Capture(LocalFleetTransport):
        async def predict(self,wid,meta,payload):
            captured.append((meta['component_id'],deepcopy(payload)))
            return await super().predict(wid,meta,payload)
    async def go():
        t=Capture(p)
        try:return await run_pipeline(p,t,lambda r,s:observation(1),tmp_path/'run',warmup_s=.06,duration_s=.24,request_timeout_s=.3)
        finally:await t.drain()
    result=asyncio.run(go());assert result['status']=='complete'
    details=[json.loads(x) for x in (tmp_path/'run/request_details.jsonl').read_text().splitlines()]
    evidence=[r['backend_evidence'] for r in details if (r['backend_evidence'] or {}).get('evidence_kind')]
    assert all(r['backend_evidence']['gpu_measurement']['reason']=='cpu_execution' for r in details if r['backend_evidence'])
    assert evidence and all(x['evidence_kind']=='synthetic_async_probe' for x in evidence)
    monitors=[q for mid,q in captured if mid=='monitor']
    assert any(json.loads(q['context']['execution_history'])['completed_chunks']>0 for q in monitors)
    assert any(q['plan'] for q in monitors)
