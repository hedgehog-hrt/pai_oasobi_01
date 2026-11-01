import asyncio,json
from copy import deepcopy
import numpy as np
import pytest
from test_global_control import spec,planner
from test_dynamic_evaluation import Sim
from rosa_repro.global_control import Supervisor
from rosa_repro.global_managers import LocalManager
from rosa_repro.global_health import watch
from rosa_repro.global_scenario import prepare
from rosa_repro.dynamic_protocol import protocol,inspect_trial,report,inspect_changes,latency_summary,sample_counts
from rosa_repro.dynamic_runner import run
from rosa_repro.global_audit import seal_tree
from rosa_repro.paired import save_state_bank,load_state_bank
from rosa_repro.regression import write_json


def make(tmp_path,changes,duration=1.5,tolerance=.2):
 s=spec(('P4',));save_state_bank(tmp_path/'bank',[np.array([1.]),np.array([2.]),np.array([3.])],task='CPU',source={'kind':'synthetic'})
 policy=protocol(s,phase='cpu_smoke',warmup_s=.2,duration_s=duration,change_tolerance_s=tolerance)
 return prepare(s,{'c0':'bank'},tmp_path,changes,warmup_s=.2,duration_s=duration,evaluation=policy,repetitions=3,
  health=dict(interval_s=.05,timeout_s=.1,max_recoveries=2))


def execute(s,root,out,rep=0,manager=None):
 async def go():
  out.mkdir();write_json(out/'scenario.json',s);write_json(out/'trial_condition.json',dict(scenario_id=s['id'],rep=rep))
  c=Supervisor(s['spec'],manager or LocalManager(),planner,out/'control');await c.start();m,states=load_state_bank(root/'bank');initial={}
  for i,(rid,cid) in enumerate(s['robots'].items()):
   case=m['cases'][(i+rep)%len(m['cases'])];initial[rid]=dict(class_id=cid,state=states[case['case_id']],seed=case['seed'])
  stop=asyncio.Event();health=asyncio.create_task(watch(c,stop,**s['health']))
  async def change(e):
   if e['op']=='join':return await c.join(e['class_id'],e['robot_id'])
   if e['op']=='leave':return await c.leave(e['robot_id'])
   if e['op']=='budget':return await c.reconcile(None,gpu_budget=e['gpu_budget'],reason='budget_change')
   wid=c.current.routes[e['robot_id']][e['component_id']];c.manager.live.fail_worker(wid)
   c.emit('owned_actor_fault_injected',worker_id=wid,epoch=c.epoch);return dict(status='injected',worker_id=wid,failed_epoch=c.epoch)
  try:
   await run(c,{r:Sim() for r in initial},initial,[(e['offset_s'],lambda e=e:change(e)) for e in s['changes']],warmup_s=s['warmup_s'],duration_s=s['duration_s'],output=out/'simulation',evaluation=s['evaluation'])
  finally:stop.set();await health;await c.close()
  seal_tree(out);return inspect_trial(out)
 return asyncio.run(go())


def test_join_rebudget_fault_recovery_leave_through_repeated_acceptance(tmp_path):
 changes=[dict(op='join',offset_s=.1,class_id='c0',robot_id='arrival'),dict(op='budget',offset_s=.4,gpu_budget=5),
  dict(op='fault',offset_s=.75,robot_id='c0-r0',component_id='monitor'),dict(op='leave',offset_s=1.1,robot_id='c0-r0')]
 s=make(tmp_path,changes);dirs={}
 for rep in range(3):
  checked=execute(s,tmp_path,tmp_path/str(rep),rep);assert checked['status']=='accepted',checked
  assert [x['op'] for x in checked['changes']]==['join','budget','fault','leave']
  assert len(checked['change_gate']['recoveries'])==1 and checked['change_gate']['recoveries'][0]['accepted']
  dirs[str(rep)]=str(rep)
 value=report(s,tmp_path,dirs);assert value['status']=='complete' and value['weighted_Q']['n']==3


def test_completion_outside_window_keeps_raw_metrics_but_rejects_trial(tmp_path):
 class SlowReplacement(LocalManager):
  def __init__(self):super().__init__();self.n=0
  async def deploy(self,spec):
   self.n+=1
   if self.n>1:await asyncio.sleep(.7)
   return await super().deploy(spec)
 s=make(tmp_path,[dict(op='join',offset_s=.15,class_id='c0',robot_id='arrival')],duration=.5)
 r=execute(s,tmp_path,tmp_path/'trial',manager=SlowReplacement())
 assert r['status']=='not_accepted' and not r['change_gate']['requirements_met']
 assert r['weighted_Q'] is None and r['raw_weighted_Q'] is not None


def journal():
 return [dict(kind='scheduled_change_started',time_ns=110,index=0,offset_s=1e-8),
  dict(kind='owned_actor_fault_injected',time_ns=111,worker_id='w',epoch=1),
  dict(kind='scheduled_change_result',time_ns=112,index=0,offset_s=1e-8,result=dict(status='injected',worker_id='w',failed_epoch=1)),
  dict(kind='health_failure',time_ns=113,worker_id='w',epoch=1),
  dict(kind='population_committed',time_ns=120,epoch=2),
  dict(kind='health_recovery_result',time_ns=121,worker_id='w',failed_epoch=1,result=dict(status='committed',epoch=2))]

@pytest.mark.parametrize('mutate',[
 lambda j:j.__delitem__(-1),
 lambda j:j[-1].update(time_ns=201),
 lambda j:j[-1].update(worker_id='different'),
 lambda j:j[-1].update(failed_epoch=0),
 lambda j:j[-1]['result'].update(status='unresolved'),
 lambda j:j.__delitem__(-2),
 lambda j:j[0].update(time_ns=99),
 lambda j:j[2].update(index=1),
 lambda j:j.append(deepcopy(j[0])),
])
def test_fault_injected_alone_or_unbound_recovery_cannot_pass(mutate):
 events=[dict(op='fault',offset_s=1e-8)];j=journal()
 assert inspect_changes(events,j,start=100,end=200,tolerance=0)[1]['requirements_met']
 mutate(j);assert not inspect_changes(events,j,start=100,end=200,tolerance=0)[1]['requirements_met']


def test_late_dispatch_and_late_completion_independently_rejected():
 e=[dict(op='join',offset_s=0)];j=[dict(kind='scheduled_change_started',time_ns=110,index=0,offset_s=0),
 dict(kind='scheduled_change_result',time_ns=120,index=0,offset_s=0,result=dict(status='committed'))]
 assert not inspect_changes(e,j,start=100,end=200,tolerance=0)[1]['requirements_met']
 assert inspect_changes(e,j,start=100,end=200,tolerance=1)[1]['requirements_met']
 j[1]['time_ns']=201;assert not inspect_changes(e,j,start=100,end=200,tolerance=1)[1]['requirements_met']

@pytest.mark.parametrize('n,eligible,p99',[(0,False,None),(999,False,None),(1000,True,989),(1001,True,990)])
def test_p99_threshold_uses_success_count_and_nearest_rank(n,eligible,p99):
 assert latency_summary(list(reversed(range(n))))==dict(successes=n,p99_ms=p99,p99_eligible=eligible)


def test_measure_sample_boundary_rejects_late_returns_and_errors():
 s=spec();base=dict(robot_id='c0-r0',component_id='s1',status='ok',invoked_ns=10,returned_ns=19)
 rows=[base,dict(base,status='error'),dict(base,returned_ns=20),dict(base,invoked_ns=9),dict(base,invoked_ns=20,returned_ns=21)]
 assert sample_counts(s,{'c0-r0':'c0'},rows,start=10,end=20,minimum=1)['requirements_met']
 assert not sample_counts(s,{'c0-r0':'c0'},rows,start=10,end=20,minimum=2)['requirements_met']


def test_repeated_fault_same_worker_binds_each_failed_generation_once():
 first=journal();second=deepcopy(first)
 for row in second:
  row['time_ns']+=50
  if 'index' in row:row.update(index=1,offset_s=6e-8)
  if 'epoch' in row:row['epoch']+=1
  if 'failed_epoch' in row:row['failed_epoch']+=1
  if 'result' in row:
   if 'epoch' in row['result']:row['result']['epoch']+=1
   if 'failed_epoch' in row['result']:row['result']['failed_epoch']+=1
 _,gate=inspect_changes([dict(op='fault',offset_s=1e-8),dict(op='fault',offset_s=6e-8)],first+second,start=100,end=200,tolerance=1e-12)
 assert gate['requirements_met'] and len(gate['recoveries'])==2
