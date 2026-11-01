import asyncio
from copy import deepcopy
import pytest
from test_prototype_validation import configs
from rosa_repro.prototype_plan import prepare,validate
from rosa_repro.dedicated_core import GroupCore
from rosa_repro.closed_loop_runtime import request_context
from rosa_repro.dummy import observation


def test_recovery_feedback_is_only_planner_input():
    outcome=dict(actual_steps=136,action_chunks=17)
    for role in ['s1','safety','monitor']:
        assert request_context(role,outcome,'previous motion rejected')==request_context(role,outcome)
    assert request_context('s2',outcome,'replan')['task_feedback']=='replan'


def concurrency_plan(limit=2,backend=2):
    c=configs();c['safety']['binding']['kwargs']['async_concurrency']=backend
    return prepare(c,{'P3':3,'P4':1},queue_policy='parallel_roles_batch_s1_v1',role_concurrency={'safety':limit})


def test_concurrent_safety_is_bounded_and_sessions_remain_separate():
    async def go():
        p=concurrency_plan();validate(p);g=GroupCore(p['workers'],p['plan_id']);core=g.cores['prototype-safety']
        original=core.backend.predict_batch;entered=0;two=asyncio.Event();release=asyncio.Event()
        async def predict(payloads):
            nonlocal entered
            entered+=1
            if entered==2:two.set()
            await release.wait();return await original(payloads)
        core.backend.predict_batch=predict;calls=[];metas=[]
        for n,rid in enumerate(core.worker['members'][:3]):
            ack=g.reset('prototype-safety','run',rid,'episode')
            m=dict(plan_id=p['plan_id'],worker_id='prototype-safety',run_id='run',robot_id=rid,episode_id='episode',request_id=str(n),attempt_id=str(n),component_id='safety',observation_seq=0,generation=ack['generation'])
            metas.append(m);calls.append(asyncio.create_task(g.predict('prototype-safety',m,{'observation':observation(n)})))
        async with asyncio.timeout(1):await two.wait()
        assert entered==2 and core.max_inflight_batches==2
        repeated=dict(metas[0],request_id='repeat',attempt_id='repeat',observation_seq=1)
        assert (await g.predict('prototype-safety',repeated,{'observation':observation(9)}))['status']=='rejected'
        calls[0].cancel()
        with pytest.raises(asyncio.CancelledError):await calls[0]
        assert core.stats()['busy_sessions']==3
        release.set();results=await asyncio.gather(*calls[1:]);await g.drain()
        assert entered==3 and core.max_inflight_batches==2 and core.stats()['busy_sessions']==0
        assert all(r['status']=='ok' and r['scheduling_evidence']['max_concurrent_invocations']==2 for r in results)
    asyncio.run(go())


def test_concurrency_cannot_exceed_backend_or_apply_to_s1():
    with pytest.raises(ValueError):GroupCore(concurrency_plan(3,2)['workers'],'plan')
    with pytest.raises(ValueError):prepare(configs(),{'P4':1},queue_policy='parallel_roles_batch_s1_v1',role_concurrency={'s1':2})
    p=concurrency_plan();p['workers'][2]['group_scheduling']['max_concurrent_invocations']=3
    with pytest.raises(ValueError):validate(p)


def test_two_process_prefetch_binds_state_and_does_not_advance_active_robot(tmp_path):
    import sys,numpy as np
    from rosa_repro.simulator_process import PrefetchedSimulator
    from rosa_repro.dummy import DummyPolicy
    async def go():
        sim=PrefetchedSimulator({'kind':'cpu_dummy'},python=sys.executable,stderr_path=tmp_path/'sim.log',timeout_s=5)
        try:
            evidence=await sim.start();assert evidence['reset_prefetch']['standby_pid']!=evidence['pid']
            current=await sim.reset(np.array([1.]),42);first_pid=sim.active.proc.pid
            sim.prepare_reset(np.array([4.]),43)
            action=DummyPolicy().get_action(current['observation'])[0];step=await sim.step(action,8)
            assert step['observation']['state.x'][0,0,0]==9 and sim.active.proc.pid==first_pid
            with pytest.raises(ValueError):await sim.reset(np.array([5.]),43)
            next_obs=await sim.reset(np.array([4.]),43)
            assert next_obs['observation']['state.x'][0,0,0]==4 and sim.active.proc.pid!=first_pid
            sim.prepare_reset(np.array([8.]),44)
        finally:await sim.close()
        assert sim.active.proc.poll() is not None and sim.standby.proc.poll() is not None
    asyncio.run(go())


def test_closed_loop_prefetch_keeps_normal_episode_accounting(tmp_path):
    import sys
    from test_prototype_validation import Local
    from test_closed_loop import suite
    from rosa_repro.prototype_plan import prepare
    from rosa_repro.closed_loop_protocol import prepare_suite,protocol
    from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
    from rosa_repro.simulator_process import PrefetchedSimulator
    s=suite(tmp_path,('P1',));c=configs();p=prepare({'s1':c['s1']},{'P1':1},frequency_hz=40,action_time_s=.001,prefetch_resets=True)
    s=prepare_suite(p,{'P1':'bank'},tmp_path,protocol(p,'cpu_smoke'),repetitions=1,setup_budget_s=20)
    async def go():
        t=Local(p);sim=PrefetchedSimulator({'kind':'cpu_dummy'},python=sys.executable,stderr_path=tmp_path/'prefetch.log',timeout_s=5)
        try:return await run_closed_loop(s,tmp_path,t,{'["P1","0"]':sim},tmp_path/'prefetched')
        finally:await sim.close();await t.group.drain()
    result=asyncio.run(go());assert result['status']=='complete',result
    assert reaudit_closed(tmp_path/'prefetched')['actions_revalidated']>0
