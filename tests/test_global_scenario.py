import asyncio,json
from copy import deepcopy
import numpy as np
import pytest
from test_global_control import spec,planner
from test_dynamic_evaluation import Sim
from rosa_repro.global_control import Supervisor
from rosa_repro.global_managers import LocalManager
from rosa_repro.global_health import watch
from rosa_repro.global_scenario import prepare,validate
from rosa_repro.global_audit import evaluate
from rosa_repro.dynamic_runner import run
from rosa_repro.paired import save_state_bank


def test_scenario_finite_state_bound_and_no_gpu_fault(tmp_path):
 save_state_bank(tmp_path/'bank',[np.array([1.])],task='CPU',source={'kind':'synthetic'})
 s=spec();events=[dict(op='join',offset_s=.1,class_id='c0',robot_id='new'),dict(op='leave',offset_s=.2,robot_id='new')]
 v=prepare(s,{'c0':'bank'},tmp_path,events,duration_s=1)
 assert validate(v,tmp_path)==v and v['robots']=={'c0-r0':'c0','new':'c0'}
 for change in ([dict(op='shell',offset_s=.1)], [dict(op='leave',offset_s=.1,robot_id='absent')], [dict(op='replan',offset_s=1)]):
  with pytest.raises(ValueError):prepare(s,{'c0':'bank'},tmp_path,change,duration_s=1)
 bad=deepcopy(v);bad['changes'][0]['robot_id']='other'
 with pytest.raises(ValueError):validate(bad,tmp_path)


def test_health_failure_triggers_global_replacement(tmp_path):
 async def go():
  manager=LocalManager();c=Supervisor(spec(),manager,planner,tmp_path/'control');await c.start();old=c.epoch
  async def fail(wid):raise RuntimeError('injected missing actor')
  manager.live.worker_stats=fail;stop=asyncio.Event();task=asyncio.create_task(watch(c,stop,interval_s=.05,timeout_s=.05,max_recoveries=1))
  for _ in range(100):
   if c.epoch>old:break
   await asyncio.sleep(.01)
  stop.set();await task;assert c.epoch==old+1 and not c.blocked
  assert any(r['kind']=='health_recovery_result' and r['result']['status']=='committed' for r in c.rows)
  await c.close()
 asyncio.run(go())


@pytest.mark.parametrize('failure',['capture','simulator'])
def test_dynamic_early_failure_keeps_fixed_window_incomplete(tmp_path,failure):
 class Broken(Sim):
  async def step(self,*args):raise RuntimeError('injected physical failure')
 async def go():
  c=Supervisor(spec(),LocalManager(),planner,tmp_path/'control');await c.start()
  sims={'c0-r0':Broken() if failure=='simulator' else Sim()};initial={'c0-r0':dict(class_id='c0',state=np.array([1.]),seed=42)}
  try:
   await run(c,sims,initial,[],warmup_s=.02,duration_s=1,output=tmp_path/'sim',max_capture_bytes=1 if failure=='capture' else 10**6)
  except ValueError:assert failure=='capture'
  await c.close();assert c.blocked
  if failure=='simulator':
   v=evaluate(tmp_path/'control');assert v['status']=='incomplete' and v['metrics'] is None
 asyncio.run(go())
