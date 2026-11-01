import asyncio,json,sys
from copy import deepcopy
import pytest
from rosa_repro.global_planner import catalog_from_plan,solve,validate_plan,ProcessPlanner
from rosa_repro.global_control import prepare,Supervisor
from rosa_repro.global_managers import LocalManager
from rosa_repro.global_audit import verify_tree
from rosa_repro.dummy import observation
from runtime_helpers import plan_for


def spec(pipelines=('P1',),timeout=1):
 cat=catalog_from_plan(plan_for(pipelines));cat['fleet']['search']['time_limit_s']=1;cat['fleet']['search']['max_iterations']=20
 return prepare(cat,{c['id']:[c['id']+'-r0'] for c in cat['fleet']['classes']},len({m['id'] for c in cat['fleet']['classes'] for m in c['components']}),call_timeout_s=timeout,deployment_timeout_s=3)

async def planner(cat,robots,budget,rev,out):return solve(cat,robots,budget,rev)
async def action(*args):await asyncio.sleep(.005);return 1


def test_population_ids_rates_and_no_fabricated_unknown_profile():
 s=spec();d=solve(s['catalog'],{'c0':['alice','bob']},1,1)
 assert d['status']=='planned' and validate_plan(d['plan'])==d['plan']
 assert {r['robot_id'] for c in d['plan']['classes'] for r in c['robots']}=={'alice','bob'}
 d['plan']['workers'][0]['members']=['forged']
 with pytest.raises(ValueError):validate_plan(d['plan'])
 cat=deepcopy(s['catalog']);cat['fleet']['profiles'][0]['points']=[dict(rate_hz=100,mean_ms=1,p99_ms=2),dict(rate_hz=101,mean_ms=1,p99_ms=2)]
 assert solve(cat,{'c0':['alice']},1,0)['status']=='unresolved'
 assert solve(s['catalog'],{'c0':[]},0,0)['status']=='idle'


def test_global_join_leave_reroutes_and_old_tickets_never_apply(tmp_path):
 async def run():
  s=spec();m=LocalManager();c=Supervisor(s,m,planner,tmp_path/'run');assert (await c.start())['status']=='committed'
  old=await c.predict('c0-r0','s1',observation(1));old_plan=c.current.plan['plan_id']
  assert (await c.join('c0','new-id'))['status']=='committed'
  assert c.current.plan['plan_id']!=old_plan and (await c.apply(old['ticket_id'],action))['status']=='rejected_stale'
  new=await c.predict('new-id','s1',observation(2));assert (await c.apply(new['ticket_id'],action))['status']=='completed'
  assert (await c.leave('c0-r0'))['status']=='committed';assert c.population=={'c0':['new-id']}
  assert m.history==[('deploy',1),('release',0),('deploy',1),('release',0),('deploy',1)]
  await c.close();assert m.live is None;verify_tree(tmp_path/'run')
 asyncio.run(run())


def test_concurrent_membership_changes_not_lost(tmp_path):
 async def run():
  c=Supervisor(spec(),LocalManager(),planner,tmp_path/'run');await c.start()
  r=await asyncio.gather(c.join('c0','a'),c.join('c0','b'));assert all(x['status']=='committed' for x in r)
  assert c.population=={'c0':['a','b','c0-r0']};await c.close()
 asyncio.run(run())


def test_rejected_plan_keeps_old_service_and_budget_failure_stops(tmp_path):
 async def run():
  c=Supervisor(spec(),LocalManager(),planner,tmp_path/'run');await c.start();old=c.current
  assert (await c.reconcile({'c0':['c0-r0','a']},gpu_budget=0))['status']=='unresolved';assert c.current is old and not c.moving
  wid=next(iter(c.current.inventory));assert (await c.recover_failure(wid,gpu_budget=0))['status']=='unresolved'
  assert c.blocked and (await c.predict('c0-r0','s1',observation(1)))['status']=='blocked';await c.close()
 asyncio.run(run())


def test_deployment_failure_rollback_uses_released_budget(tmp_path):
 class FailOnce(LocalManager):
  def __init__(self):super().__init__();self.n=0
  async def deploy(self,s):
   self.n+=1
   if self.n==2:raise RuntimeError('injected deployment failure')
   return await super().deploy(s)
 async def run():
  m=FailOnce();c=Supervisor(spec(),m,planner,tmp_path/'run');await c.start()
  assert (await c.join('c0','new'))['status']=='rolled_back';assert c.population=={'c0':['c0-r0']} and not c.blocked
  v=await c.predict('c0-r0','s1',observation(1));assert v['status']=='ok';await c.close()
 asyncio.run(run())


@pytest.mark.parametrize('forced',[False,True])
def test_inflight_action_unknown_blocks_redeployment(tmp_path,forced):
 async def run():
  c=Supervisor(spec(timeout=.02),LocalManager(),planner,tmp_path/'run');await c.start()
  v=await c.predict('c0-r0','s1',observation(1))
  async def slow(*args):await asyncio.sleep(.2);return 1
  t=asyncio.create_task(c.apply(v['ticket_id'],slow));await asyncio.sleep(.005)
  change=c.recover_failure(next(iter(c.current.inventory))) if forced else c.join('c0','new')
  assert (await change)['status']=='blocked_unknown_work';assert await t==dict(status='unknown',requires_external_recovery_review=True)
  await asyncio.sleep(.21);await c.close();assert json.loads((tmp_path/'run/control_result.json').read_text())['status']=='incomplete'
 asyncio.run(run())


def test_process_planner_is_current_source_bound(tmp_path):
 s=spec();p=ProcessPlanner(sys.executable);d=asyncio.run(p(s['catalog'],s['initial_population'],1,1,tmp_path/'solver'))
 assert d['status']=='planned' and (tmp_path/'solver/process.json').exists()


def test_dead_actor_pending_rpc_is_invalidated_before_confirmed_release(tmp_path):
 async def run():
  c=Supervisor(spec(timeout=.03),LocalManager(),planner,tmp_path/'run');await c.start();old=c.current
  async def dead(*args):await asyncio.Event().wait()
  old.transport.predict=dead
  request=asyncio.create_task(c.predict('c0-r0','s1',observation(1)));await asyncio.sleep(.005)
  assert (await c.recover_failure(next(iter(old.inventory))))['status']=='committed'
  assert (await request)['status']=='timeout' and not old.pending and old.closed
  assert any(r['kind']=='invalidated_model_rpcs' for r in old.events)
  assert (await c.predict('c0-r0','s1',observation(1)))['status']=='ok';await c.close()
 asyncio.run(run())


def test_new_robot_cannot_dispatch_a_slot_before_membership_commit(tmp_path):
 async def run():
  c=Supervisor(spec(('P2',)),LocalManager(),planner,tmp_path/'run');await c.start()
  await c.join('c0','arrival');reset=c.current.state['arrival']['start_ns'];commit=c.admitted_since['arrival']
  assert reset<commit
  before=len(c.current.events)
  result=await c.predict('arrival','monitor',observation(1),scheduled_ns=reset,slot_index=0)
  assert result['status']=='slot_before_epoch' and len(c.current.events)==before
  assert (await c.predict('arrival','monitor',observation(1),scheduled_ns=commit,slot_index=1))['status']=='ok'
  await c.close()
 asyncio.run(run())
