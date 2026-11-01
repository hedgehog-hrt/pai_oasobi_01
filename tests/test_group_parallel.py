import asyncio
from copy import deepcopy
import pytest
from test_group_deadlines import setup
from test_prototype_validation import configs
from rosa_repro.prototype_plan import prepare,validate
from rosa_repro.dummy import observation


def test_parallel_roles_overlap_but_same_model_stays_serialized():
    async def go():
        group,metas=setup('parallel_roles_v1');release=asyncio.Event();entered=set();all_entered=asyncio.Event()
        for core in group.cores.values():
            original=core.backend.predict_batch;role=core.worker['role']
            async def predict(payloads,original=original,role=role):
                entered.add(role)
                if len(entered)==4:all_entered.set()
                await release.wait();return original(payloads)
            core.backend.predict_batch=predict
        calls=[asyncio.create_task(group.predict(m['worker_id'],m,{'observation':observation(1)})) for m in metas.values()]
        async with asyncio.timeout(1):await all_entered.wait()
        assert group.peak==4
        core=group.cores['prototype-monitor'];m=metas['monitor']
        # A new episode replaces the session while the old inference is outstanding.
        ack=group.reset(m['worker_id'],m['run_id'],m['robot_id'],'next-episode')
        next_meta=dict(m,episode_id='next-episode',generation=ack['generation'],request_id='next',attempt_id='next')
        next_call=asyncio.create_task(group.predict(m['worker_id'],next_meta,{'observation':observation(2)}))
        await asyncio.sleep(.01)
        assert core.max_inflight_batches==1 and core.model_calls==1
        release.set();results=await asyncio.gather(*calls,next_call);await group.drain()
        assert results[-1]['status']=='ok' and results[3]['status']=='cancelled'
        assert core.max_inflight_batches==1 and group.active==0
        assert all(c.stats()['busy_sessions']==0 for c in group.cores.values())
        assert any(r['scheduling_evidence']['other_active_invocations_at_dispatch']>0 for r in results)
    asyncio.run(go())


def test_cancelled_client_does_not_release_running_model_or_block_other_roles():
    async def go():
        group,metas=setup('parallel_roles_v1');entered=asyncio.Event();release=asyncio.Event()
        core=group.cores['prototype-monitor'];original=core.backend.predict_batch
        async def predict(payloads):entered.set();await release.wait();raise RuntimeError('backend failed')
        core.backend.predict_batch=predict
        m=metas['monitor'];call=asyncio.create_task(group.predict(m['worker_id'],m,{'observation':observation(1)}))
        await entered.wait();call.cancel()
        with pytest.raises(asyncio.CancelledError):await call
        s=metas['s1']
        async with asyncio.timeout(1):r=await group.predict(s['worker_id'],s,{'observation':observation(1)})
        assert r['status']=='ok' and core.stats()['busy_sessions']==1
        release.set();await group.drain();assert core.stats()['busy_sessions']==0
        core.backend.predict_batch=original
        next_meta=dict(m,request_id='next',attempt_id='next',observation_seq=1)
        assert (await group.predict(m['worker_id'],next_meta,{'observation':observation(1)}))['status']=='ok'
    asyncio.run(go())


def test_parallel_policy_is_bound_and_excludes_p1():
    c=configs();p=prepare(c,{'P4':1},queue_policy='parallel_roles_v1');validate(p)
    assert not p['R07_profile_adoption'] and not p['paper_model_placement']
    bad=deepcopy(p);bad['workers'][0]['group_scheduling']['policy']='fifo'
    with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):prepare({'s1':c['s1']},{'P1':1},queue_policy='parallel_roles_v1')


def test_parallel_closed_loop_preserves_action_and_raw_audit(tmp_path):
    import json
    import numpy as np
    from test_prototype_validation import Local
    from test_closed_loop import Sim
    from rosa_repro.paired import save_state_bank
    from rosa_repro.closed_loop_protocol import protocol,prepare_suite
    from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
    values=configs()
    for c in values.values():c['binding']['kwargs']['delay_s']=.015
    plan=prepare(values,{'P4':2},frequency_hz=20,action_time_s=.001,queue_policy='parallel_roles_v1')
    save_state_bank(tmp_path/'bank',[np.array([1.,.3])],task='CPU',source={'kind':'synthetic'})
    pro=protocol(plan,'cpu_smoke',warmup_s=.1,duration_s=2.8,request_timeout_s=2,drain_timeout_s=2)
    suite=prepare_suite(plan,{'P4':'bank'},tmp_path,pro,repetitions=1,setup_budget_s=10)
    async def go():
        t=Local(plan);sims={r['robot_id']:Sim() for r in plan['classes'][0]['robots']}
        result=await run_closed_loop(suite,tmp_path,t,sims,tmp_path/'run');await t.group.drain()
        assert t.group.peak>1 and all(c.max_inflight_batches==1 for c in t.group.cores.values())
        return result
    result=asyncio.run(go());assert result['status']=='complete'
    audit=reaudit_closed(tmp_path/'run');assert audit['actions_revalidated']>0
    rows=[json.loads(x) for x in (tmp_path/'run/request_details.jsonl').read_text().splitlines()]
    assert all(x['scheduling_evidence']['policy']=='parallel_roles_v1' for x in rows)
