"""Three clients with capacity two must preserve the partially filled tail."""
import asyncio
import numpy as np
from test_prototype_validation import configs
from rosa_repro.dedicated_core import GroupCore
from rosa_repro.dummy import observation
from rosa_repro.prototype_plan import prepare


def test_three_clients_are_two_plus_one_without_aliasing_or_dropping_tail():
    async def run():
        values=configs();values['s1'].update(batch_size=2,batch_wait_ms=2)
        plan=prepare(values,{'P2':1,'P3':1,'P4':1},frequency_hz=2,action_time_s=.4,
            queue_policy='parallel_roles_batch_s1_v1',periodic_phases='spread_classes_v1')
        group=GroupCore(plan['workers'],plan['plan_id']);core=group.cores['prototype-s1']
        calls=[];metadata=[]
        for i,rid in enumerate(core.worker['members']):
            ack=group.reset('prototype-s1','run',rid,'episode')
            meta=dict(plan_id=plan['plan_id'],worker_id='prototype-s1',run_id='run',robot_id=rid,
                episode_id='episode',request_id=str(i),attempt_id=str(i),component_id='s1',observation_seq=0,generation=ack['generation'])
            metadata.append(meta)
            calls.append(group.predict('prototype-s1',meta,{'observation':observation(i+1)}))
        async with asyncio.timeout(2):
            results=await asyncio.gather(*calls);await group.drain()
        assert core.batch_sizes==[2,1]
        assert [r['actual_batch_size'] for r in results]==[2,2,1]
        assert results[0]['batch_id']==results[1]['batch_id']!=results[2]['batch_id']
        for i,result in enumerate(results):
            assert result['status']=='ok' and result['metadata']==metadata[i]
            assert np.all(result['output']['action.x']==i+1)
        assert core.max_inflight_batches==1 and core.stats()['busy_sessions']==0
        assert not group.batch_pumps and group.active==0
    asyncio.run(run())
