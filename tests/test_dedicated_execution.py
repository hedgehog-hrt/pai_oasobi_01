import asyncio,os
from copy import deepcopy
import numpy as np
import pytest
from runtime_helpers import plan_for
from rosa_repro.dedicated_baselines import prepare
from rosa_repro.dedicated_execution import compile_plan,validate,check_inventory,launch_gate
from rosa_repro.dedicated_core import GroupCore
from rosa_repro.closed_loop_protocol import protocol,prepare_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.paired import save_state_bank
from rosa_repro.dummy import observation
from test_closed_loop import Sim


def plan(method='gpu_per_robot',pipelines=('P4',),budget=10):
 p=plan_for(pipelines);return compile_plan(prepare(p,method,{w['component_id']:w['profile_id'] for w in p['workers']},gpu_budget=budget))


@pytest.mark.parametrize('method,groups',[('gpu_per_robot',2),('gpu_per_model_per_robot',5)])
def test_explicit_groups_entire_workload_uncapped(method,groups):
 p=plan(method,('P4','P1'));assert len(p['physical_groups'])==groups and len(p['workers'])==5
 assert all(c['frequency_hz'] is None for c in p['classes']) and all(w['batch_size']==1 and len(w['members'])==1 for w in p['workers'])
 assert validate(p)==p
 q=deepcopy(p);q['physical_groups'][0]['resources']['num_gpus']=.25
 with pytest.raises(ValueError):validate(q)
 with pytest.raises(ValueError):plan(method,('P4','P1'),budget=1)


def test_group_inventory_accepts_real_colocation_rejects_aliases():
 p=plan();g=GroupCore(p['workers'],p['plan_id'])
 inv={w:dict(g.stats(w),node_id='test',pid=os.getpid(),gpu_ids=[]) for w in g.cores}
 assert check_inventory(p,inv)['physical_groups']==1
 bad=deepcopy(inv);bad[next(iter(bad))]['pid']+=1
 with pytest.raises(ValueError):check_inventory(p,bad)
 bad=deepcopy(inv);values=list(bad.values());values[1]['instance_id']=values[0]['instance_id']
 with pytest.raises(ValueError):check_inventory(p,bad)


def test_group_serialization_timeout_does_not_release_busy_core():
 async def run():
  p=plan();g=GroupCore(p['workers'],p['plan_id']);tasks=[]
  for wid,c in g.cores.items():
   original=c.backend.predict_batch
   async def slow(payloads,original=original):await asyncio.sleep(.03);return original(payloads)
   c.backend.predict_batch=slow;rid=c.worker['members'][0];g.reset(wid,'run',rid,'ep')
   meta=dict(run_id='run',robot_id=rid,episode_id='ep',generation=0,request_id=wid,attempt_id='a',observation_seq=0,component_id=c.worker['component_id'],worker_id=wid,plan_id=p['plan_id'])
   tasks.append(asyncio.create_task(g.predict(wid,meta,dict(observation=observation(1)))))
  await asyncio.sleep(.005);tasks[0].cancel();await asyncio.gather(*tasks,return_exceptions=True);await g.drain()
  assert g.peak==1 and all(not c.stats()['busy_sessions'] for c in g.cores.values()) and all(c.model_calls==1 for c in g.cores.values())
 asyncio.run(run())


def test_co_resident_p4_closed_loop_reaudits(tmp_path):
 p=plan();save_state_bank(tmp_path/'bank',[np.array([1.]),np.array([2.])],task='CPU',source={'kind':'synthetic'})
 s=prepare_suite(p,{'c0':'bank'},tmp_path,protocol(p,'cpu_smoke',warmup_s=.1,duration_s=.5),repetitions=1)
 class Local:
  def __init__(self):self.group=GroupCore(p['workers'],p['plan_id']);self.launch_gate=launch_gate(p)
  async def reset(self,w,*a):return self.group.reset(w,*a)
  async def predict(self,w,*a):return await self.group.predict(w,*a)
  async def inventory(self):return {w:dict(self.group.stats(w),node_id='local-fixture',pid=os.getpid(),gpu_ids=[]) for w in self.group.cores}
 async def run():
  t=Local();r=await run_closed_loop(s,tmp_path,t,{p['classes'][0]['robots'][0]['robot_id']:Sim()},tmp_path/'run');await t.group.drain();return r
 assert asyncio.run(run())['status']=='complete'
 assert reaudit_closed(tmp_path/'run')['actions_revalidated']>0


def test_cpu_cannot_become_joint_gpu_evidence(tmp_path):
 from rosa_repro.dedicated_residency import inspect
 p=plan()
 with pytest.raises(ValueError):inspect(p,tmp_path,['a','b','c'])
 with pytest.raises(ValueError):launch_gate(p,joint_review={'approved':True})


def test_gpu_group_requires_joint_review_or_explicit_calibration():
 from rosa_repro.execution_plan import compile_execution_plan
 ref=plan_for();x=ref['inputs'];ref=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings={m:dict(factory='test:backend',kwargs={}) for m in x['roles']})
 p=compile_plan(prepare(ref,'gpu_per_robot',{w['component_id']:w['profile_id'] for w in ref['workers']},gpu_budget=1))
 with pytest.raises(ValueError):launch_gate(p)
 assert launch_gate(p,calibration=True)==dict(mode='GPU_joint_residency_calibration',comparison_eligible=False)


@pytest.mark.parametrize('method,count',[('gpu_per_robot',1),('gpu_per_model_per_robot',4)])
def test_gpu_lease_validator_with_explicit_synthetic_inventory(method,count):
 from rosa_repro.execution_plan import compile_execution_plan,digest,runtime_source_manifest
 ref=plan_for();x=ref['inputs'];ref=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings={m:dict(factory='test:backend',kwargs={}) for m in x['roles']})
 p=compile_plan(prepare(ref,method,{w['component_id']:w['profile_id'] for w in ref['workers']},gpu_budget=4));rows={}
 for w in p['workers']:
  index=next(i for i,g in enumerate(p['physical_groups']) if w['worker_id'] in g['worker_ids'])
  rows[w['worker_id']]=dict(plan_id=p['plan_id'],worker_spec_sha256=digest(w),runtime_source_sha256=runtime_source_manifest()['sha256'],
   physical_group_id=w['physical_group_id'],instance_id=w['worker_id'],node_id='SYNTHETIC_UNIT_FIXTURE',pid=index+100,gpu_ids=[str(index)],
   execution_kind='external_model',gpu_properties=dict(name=w['hardware_kind'],memory_mib=100000))
 assert check_inventory(p,rows)['unique_gpu_leases']==count
 bad=deepcopy(rows);values=list(bad.values())
 if count==1:values[0]['gpu_ids']=['another']
 else:values[1]['gpu_ids']=values[0]['gpu_ids']
 with pytest.raises(ValueError):check_inventory(p,bad)


def test_conflicting_vllm_memory_reservations_rejected_before_loading():
 from rosa_repro.execution_plan import compile_execution_plan
 ref=plan_for();x=ref['inputs'];bindings={m:dict(factory='test:backend',kwargs=dict(config=dict(engine='vllm',gpu_memory_utilization=.8))) for m in x['roles']}
 ref=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings=bindings)
 resource=prepare(ref,'gpu_per_robot',{w['component_id']:w['profile_id'] for w in ref['workers']},gpu_budget=1)
 with pytest.raises(ValueError,match='headroom'):compile_plan(resource)
