import asyncio
from copy import deepcopy
import json
import numpy as np
import pytest
from test_prototype_validation import configs,Local
from test_closed_loop import Sim
from rosa_repro.dedicated_core import GroupCore
from rosa_repro.prototype_plan import prepare,validate
from rosa_repro.dummy import observation


def batch_plan():
    values=configs();values['s1'].update(batch_size=3,batch_wait_ms=10)
    return prepare(values,{'P2':1,'P3':1,'P4':1},queue_policy='parallel_roles_batch_s1_v1',periodic_phases='spread_classes_v1')


def setup():
    p=batch_plan();g=GroupCore(p['workers'],p['plan_id']);metas=[];w=g.cores['prototype-s1'].worker
    for i,rid in enumerate(w['members']):
        ack=g.reset(w['worker_id'],'run',rid,'e0')
        metas.append(dict(plan_id=p['plan_id'],worker_id=w['worker_id'],run_id='run',robot_id=rid,
            episode_id='e0',request_id=str(i),attempt_id=str(i),component_id='s1',observation_seq=0,generation=ack['generation']))
    return g,metas


def test_batch_split_preserves_client_identity_and_one_inflight_call():
    async def go():
        g,ms=setup()
        rows=await asyncio.gather(*(g.predict('prototype-s1',m,{'observation':observation(i+1)}) for i,m in enumerate(ms)))
        await g.drain()
        assert len({r['batch_id'] for r in rows})==1
        for i,r in enumerate(rows):
            assert r['metadata']==ms[i] and r['actual_batch_size']==3 and r['status']=='ok'
            assert np.all(r['output']['action.x']==i+1)
            assert r['scheduling_evidence']['queue_wait_ms']>=0
        c=g.cores['prototype-s1'];assert c.model_calls==1 and c.max_inflight_batches==1 and c.stats()['busy_sessions']==0
    asyncio.run(go())


def test_partial_batch_flushes_and_failure_releases_all_sessions():
    async def go():
        g,ms=setup();c=g.cores['prototype-s1'];c.backend.fail_calls={1}
        async with asyncio.timeout(1):row=await g.predict('prototype-s1',ms[0],{'observation':observation(1)})
        assert row['status']=='error' and row['actual_batch_size']==1
        m=dict(ms[0],request_id='retry',attempt_id='retry',observation_seq=1)
        assert (await g.predict('prototype-s1',m,{'observation':observation(1)}))['status']=='ok'
        await g.drain();assert g.active==0 and c.stats()['busy_sessions']==0
    asyncio.run(go())


def test_cancelled_caller_and_reset_never_release_running_batch():
    async def go():
        g,ms=setup();c=g.cores['prototype-s1'];entered=asyncio.Event();release=asyncio.Event();original=c.backend.predict_batch
        async def block(payloads):entered.set();await release.wait();return original(payloads)
        c.backend.predict_batch=block
        calls=[asyncio.create_task(g.predict('prototype-s1',m,{'observation':observation(i)})) for i,m in enumerate(ms)]
        await entered.wait();calls[0].cancel()
        with pytest.raises(asyncio.CancelledError):await calls[0]
        ack=g.reset('prototype-s1','run',ms[1]['robot_id'],'e1')
        m=dict(ms[1],episode_id='e1',generation=ack['generation'],request_id='next',attempt_id='next')
        queued=asyncio.create_task(g.predict('prototype-s1',m,{'observation':observation(9)}))
        await asyncio.sleep(.02);assert c.model_calls==1 and c.stats()['busy_sessions']==3
        release.set();rows=await asyncio.gather(*calls[1:],queued);await g.drain()
        assert rows[0]['status']=='cancelled' and rows[-1]['status']=='ok'
        assert np.all(rows[-1]['output']['action.x']==9)
        assert c.max_inflight_batches==1 and c.stats()['busy_sessions']==0 and not g.batch_pumps
    asyncio.run(go())


def test_phase_policy_changes_only_offsets_and_is_plan_bound():
    p=batch_plan();validate(p)
    expected={'P2':{'monitor':0},'P3':{'safety':0,'monitor':666666666},'P4':{'safety':250000000,'monitor':1333333333}}
    for c in p['classes']:
        for m in c['audit']['components']:
            assert m['deadline_ms']=={'s1':200,'safety':500,'s2':2000,'monitor':2000}[m['role']]
            if 'period_ns' in m:
                assert m['phase_ns']==expected[c['audit']['id']][m['role']]
                assert m['period_ns']=={'safety':500000000,'monitor':2000000000}[m['role']]
    bad=deepcopy(p);bad['classes'][-1]['audit']['components'][-1]['phase_ns']=0
    with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):prepare(configs(),{'P4':1},periodic_phases='unsupported')
    c=configs();c['s1'].update(batch_size=3,batch_wait_ms=11)
    with pytest.raises(ValueError):prepare(c,{'P4':3},queue_policy='parallel_roles_batch_s1_v1')


def test_batch_phases_preserve_closed_loop_raw_audit_and_periodic_schedule(tmp_path):
    from rosa_repro.paired import save_state_bank
    from rosa_repro.closed_loop_protocol import protocol,prepare_suite
    from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
    values=configs();values['s1'].update(batch_size=3,batch_wait_ms=10)
    plan=prepare(values,{'P2':1,'P3':1,'P4':1},frequency_hz=20,action_time_s=.001,
        queue_policy='parallel_roles_batch_s1_v1',periodic_phases='spread_classes_v1')
    save_state_bank(tmp_path/'bank',[np.array([1.,.3])],task='CPU',source={'kind':'synthetic'})
    pro=protocol(plan,'cpu_smoke',warmup_s=.1,duration_s=2.8,request_timeout_s=2,drain_timeout_s=2)
    suite=prepare_suite(plan,{c:'bank' for c in ['P2','P3','P4']},tmp_path,pro,repetitions=1,setup_budget_s=10)
    async def go():
        t=Local(plan);sims={r['robot_id']:Sim() for c in plan['classes'] for r in c['robots']}
        result=await run_closed_loop(suite,tmp_path,t,sims,tmp_path/'run');await t.group.drain()
        assert all(c.max_inflight_batches==1 for c in t.group.cores.values())
        assert max(t.group.cores['prototype-s1'].batch_sizes)>1
        return result
    result=asyncio.run(go());assert result['status']=='complete'
    audit=reaudit_closed(tmp_path/'run');assert audit['actions_revalidated']>0
    rows=[json.loads(x) for x in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    # The raw auditor independently binds phase/slot/epoch and every action tensor.
    assert any(r['record_type']=='request' and r['component_id']=='safety' and r['slot_index'] is not None for r in rows)
