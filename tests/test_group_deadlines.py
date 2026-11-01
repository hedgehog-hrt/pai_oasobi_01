import asyncio
from copy import deepcopy
import pytest
from test_prototype_validation import configs
from rosa_repro.prototype_plan import prepare,validate
from rosa_repro.dedicated_core import GroupCore,DeadlineGate
from rosa_repro.dummy import observation


def setup(policy='edf_queue_expiry_v1'):
    plan=prepare(configs(),{'P4':1},queue_policy=policy)
    group=GroupCore(plan['workers'],plan['plan_id']);metas={}
    for worker in plan['workers']:
        wid=worker['worker_id'];rid=worker['members'][0];ack=group.reset(wid,'run',rid,'episode')
        metas[worker['role']]=dict(plan_id=plan['plan_id'],worker_id=wid,run_id='run',robot_id=rid,
            episode_id='episode',request_id=wid,attempt_id=wid,component_id=worker['role'],observation_seq=0,generation=ack['generation'])
    return group,metas


@pytest.mark.parametrize('policy,expected',[('edf_queue_expiry_v1',['monitor','s1','safety','s2']),('fifo',['monitor','s2','safety','s1'])])
def test_queued_urgency_reorders_without_overlapping_or_preempting(policy,expected):
    async def go():
        group,metas=setup(policy);entered=asyncio.Event();release=asyncio.Event();order=[]
        for core in group.cores.values():
            original=core.backend.predict_batch;role=core.worker['role']
            async def predict(payloads,original=original,role=role):
                order.append(role)
                if role=='monitor':entered.set();await release.wait()
                return original(payloads)
            core.backend.predict_batch=predict
        async def request(role):return await group.predict(metas[role]['worker_id'],metas[role],{'observation':observation(1)})
        first=asyncio.create_task(request('monitor'));await entered.wait()
        others=[asyncio.create_task(request(role)) for role in ['s2','safety','s1']]
        await asyncio.sleep(.005);assert order==['monitor'];release.set()
        rs=await asyncio.gather(first,*others);await group.drain()
        assert order==expected and group.peak==1 and all(r['status']=='ok' for r in rs)
        assert all(not r['scheduling_evidence']['expired_before_inference'] for r in rs)
    asyncio.run(go())


def test_expired_safety_is_not_inferred_or_returned_safe_and_session_recovers():
    async def go():
        group,metas=setup();group.cores['prototype-safety'].worker['group_scheduling']['queue_budget_ms']=1
        core=group.cores['prototype-monitor'];original=core.backend.predict_batch
        entered=asyncio.Event();release=asyncio.Event()
        async def block(payloads):entered.set();await release.wait();return original(payloads)
        core.backend.predict_batch=block
        async def request(role,meta=None):return await group.predict(metas[role]['worker_id'],meta or metas[role],{'observation':observation(1)})
        first=asyncio.create_task(request('monitor'));await entered.wait()
        safety=asyncio.create_task(request('safety'));await asyncio.sleep(.01);release.set()
        _,r=await asyncio.gather(first,safety)
        assert r['status']=='error' and r['error_type']=='QueueDeadlineExceeded' and r['output'] is None
        stats=group.stats('prototype-safety');assert stats['model_calls']==0 and stats['busy_sessions']==0 and stats['queue_expired_requests']==1
        meta=dict(metas['safety'],request_id='next',attempt_id='next',observation_seq=1)
        assert (await request('safety',meta))['status']=='ok'
    asyncio.run(go())


def test_cancelled_grantee_does_not_deadlock_gate():
    async def go():
        gate=DeadlineGate();await gate.acquire(1)
        waiting=asyncio.create_task(gate.acquire(2));await asyncio.sleep(0)
        gate.release();waiting.cancel()
        with pytest.raises(asyncio.CancelledError):await waiting
        async with asyncio.timeout(.2):await gate.acquire(3)
        gate.release()
    asyncio.run(go())


def test_policy_and_role_budgets_are_bound_to_plan():
    p=prepare(configs(),{'P4':1},queue_policy='edf_queue_expiry_v1');validate(p)
    bad=deepcopy(p);bad['workers'][0]['group_scheduling']['queue_budget_ms']=999
    with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):prepare({'s1':configs()['s1']},{'P1':1},queue_policy='edf_queue_expiry_v1')


def test_closed_loop_records_expiry_without_executing_expired_actions(tmp_path):
    import json
    import numpy as np
    from test_prototype_validation import Local
    from test_closed_loop import Sim
    from rosa_repro.paired import save_state_bank
    from rosa_repro.closed_loop_protocol import protocol,prepare_suite
    from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
    values=configs();values['monitor']['binding']['kwargs']['delay_s']=.3
    plan=prepare(values,{'P4':1},frequency_hz=40,action_time_s=.001,queue_policy='edf_queue_expiry_v1')
    save_state_bank(tmp_path/'bank',[np.array([1.,.3])],task='CPU',source={'kind':'synthetic'})
    # Monitor's first global slot is at 2s; cover it and allow the injected wait.
    pro=protocol(plan,'cpu_smoke',warmup_s=.1,duration_s=2.8,request_timeout_s=4,drain_timeout_s=4)
    suite=prepare_suite(plan,{'P4':'bank'},tmp_path,pro,repetitions=1,setup_budget_s=10)
    class SlowMonitorFirst(Local):
        def __init__(self,plan):
            super().__init__(plan);self.monitor_started=asyncio.Event();self.first_s1=True
            core=self.group.cores['prototype-monitor'];original=core.backend.predict_batch
            async def monitor(payloads):
                self.monitor_started.set()
                return await asyncio.to_thread(original,payloads)
            core.backend.predict_batch=monitor
        async def predict(self,wid,*args):
            if wid=='prototype-s1' and self.first_s1:
                self.first_s1=False;await self.monitor_started.wait()
            return await super().predict(wid,*args)
    async def go():
        t=SlowMonitorFirst(plan)
        result=await run_closed_loop(suite,tmp_path,t,{plan['classes'][0]['robots'][0]['robot_id']:Sim()},tmp_path/'run')
        await t.group.drain();return result
    result=asyncio.run(go())
    assert result['drained'] and result.get('error_type')!='TimeoutError'
    details=[json.loads(x) for x in (tmp_path/'run/request_details.jsonl').read_text().splitlines()]
    expired={x['request_id'] for x in details if x['error_type']=='QueueDeadlineExceeded'}
    assert expired and all(x['scheduling_evidence']['expired_before_inference'] for x in details if x['request_id'] in expired)
    events=[json.loads(x) for x in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    assert not any(x['record_type']=='action' and x['request_id'] in expired for x in events)
    if result['status']=='complete':reaudit_closed(tmp_path/'run')
