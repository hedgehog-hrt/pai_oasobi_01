import asyncio,json
from copy import deepcopy
import pytest
from rosa_repro.dynamic_control import prepare,admission,Controller,reaudit
from rosa_repro.control_local import LocalControlTransport
from rosa_repro.dummy import observation
from runtime_helpers import plan_for
from rosa_repro.execution_plan import DUMMY_FACTORY


def design(*,delay=.005,timeout=1.,pipeline='P1'):
 bindings={role:dict(factory=DUMMY_FACTORY,kwargs=dict(delay_s=delay)) for role in {'P1':['s1'],'P4':['s1','s2','safety','monitor']}[pipeline]}
 p=plan_for((pipeline,),bindings=bindings);wid=p['workers'][0]['worker_id'];rid=p['classes'][0]['robots'][0]['robot_id']
 return prepare(p,[rid],{wid:'standby'},physical_gpu_budget=len(p['workers'])+1,timeout_s=timeout),rid,wid


async def action(rid,value,duration):await asyncio.sleep(duration);return 1


def test_resource_budget_and_unprofiled_admission():
 s,rid,wid=design()
 with pytest.raises(ValueError):prepare(s['blueprint'],[rid],{wid:'standby'},physical_gpu_budget=len(s['blueprint']['workers']))
 assert admission(s,{rid})['status']=='admitted'
 assert admission(s,{'unconfigured'})['status']=='requires_global_replan'


def test_inflight_old_reply_discarded_and_standby_applies_once(tmp_path):
 async def go():
  s,rid,wid=design(delay=.08);t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  pending=asyncio.create_task(c.predict(rid,'s1',observation(1)));await asyncio.sleep(.02)
  t.fail_worker(wid);assert (await c.healthcheck())[wid]=='failed'
  switched=await c.failover(wid);assert switched['status']=='ready'
  assert (await pending)['status']=='stale'
  value=await c.predict(rid,'s1',observation(2));assert value['status']=='ok'
  assert (await c.apply(value['ticket_id'],action))['status']=='completed'
  assert (await c.apply(value['ticket_id'],action))['status']=='rejected_duplicate_or_unknown'
  result=await c.close();await t.drain();return result
 r=asyncio.run(go());assert r['status']=='complete' and r['stale_responses']==1 and r['actions_completed']==1
 assert r['failovers'][0]['ready_after_ns'] is not None and reaudit(tmp_path/'trace')==r


def test_delivered_action_invalidated_by_leave_and_rejoin(tmp_path):
 async def go():
  s,rid,wid=design();t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  v=await c.predict(rid,'s1',observation(1));assert (await c.leave(rid))['status']=='left'
  assert (await c.join(rid))['status']=='admitted'
  assert (await c.apply(v['ticket_id'],action))['status']=='rejected_stale'
  assert (await c.join('other'))['status']=='requires_global_replan'
  return await c.close()
 assert asyncio.run(go())['actions_started']==0


def test_standby_restarted_or_missing_never_resumes(tmp_path):
 async def go():
  s,rid,wid=design();t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  t.cores['standby'].instance_id='restarted'
  assert (await c.failover(wid))['status']=='blocked_standby_validation'
  assert (await c.predict(rid,'s1',observation(1)))['status']=='blocked'
  return await c.close()
 assert asyncio.run(go())['actions_started']==0


def test_model_timeout_late_completion_never_creates_ticket(tmp_path):
 async def go():
  s,rid,wid=design(delay=.08,timeout=.02);t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  v=await c.predict(rid,'s1',observation(1));assert v['status']=='timeout'
  await t.drain();await asyncio.sleep(0);return await c.close()
 r=asyncio.run(go());assert r['actions_started']==0
 assert 'discarded_late_response' in (tmp_path/'trace/events.jsonl').read_text()


def test_unknown_action_blocks_failover_and_rejoin(tmp_path):
 async def go():
  s,rid,wid=design(timeout=.03);t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  v=await c.predict(rid,'s1',observation(1))
  async def slow(*args):await asyncio.sleep(.15);return 1
  applied=asyncio.create_task(c.apply(v['ticket_id'],slow));await asyncio.sleep(.01)
  assert (await c.failover(wid))['status']=='blocked_unknown_action'
  assert (await applied)['status']=='unknown'
  assert (await c.predict(rid,'s1',observation(2)))['status']=='blocked'
  await asyncio.sleep(.17);return await c.close()
 r=asyncio.run(go());assert r['status']=='incomplete' and r['unknown_actions']==1 and r['actions_completed']==0


def test_plan_cache_and_safety_reset_on_failover(tmp_path):
 async def go():
  s,rid,wid=design(pipeline='P4');t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  assert (await c.predict(rid,'s1',observation(1)))['status']=='needs_plan'
  assert (await c.predict(rid,'s2',observation(1)))['status']=='ok'
  await c.predict(rid,'safety',observation(1));v=await c.predict(rid,'s1',observation(1));assert v['status']=='ok'
  assert (await c.apply(v['ticket_id'],action))['status']=='completed'
  await c.failover(wid)
  assert c.state[rid]['plan'] is None and c.state[rid]['safety'] is None
  assert (await c.predict(rid,'s1',observation(1)))['status']=='needs_plan'
  return await c.close()
 r=asyncio.run(go());assert not r['full_P2_P4_qualified_metric_claim']


def test_superseded_ticket_and_tamper_rejected(tmp_path):
 async def go():
  s,rid,wid=design();t=LocalControlTransport(s);c=Controller(s,t,tmp_path/'trace');await c.start()
  old=await c.predict(rid,'s1',observation(1));await asyncio.sleep(.04)
  new=await c.predict(rid,'s1',observation(2))
  assert (await c.apply(old['ticket_id'],action))['status']=='rejected_stale'
  assert (await c.apply(new['ticket_id'],action))['status']=='completed'
  return await c.close()
 asyncio.run(go());p=tmp_path/'trace/events.jsonl';p.write_text(p.read_text().replace('action_finished','forged_action'))
 with pytest.raises(ValueError,match='evidence changed'):reaudit(tmp_path/'trace')


def test_scenario_rejects_undeclared_steps():
 from rosa_repro.control_scenario import validate_scenario
 s,rid,wid=design()
 with pytest.raises(ValueError):validate_scenario(s,[dict(op='wait',seconds=float('inf'))])
 with pytest.raises(ValueError):validate_scenario(s,[dict(op='apply',ticket='absent',expect='completed')])
 with pytest.raises(ValueError):validate_scenario(s,[dict(op='predict',robot=rid,component='other',**{'as':'a'},expect='ok')])
