import asyncio,json,time
from copy import deepcopy
import numpy as np
import pytest
from rosa_repro.global_control import Supervisor
from rosa_repro.global_managers import LocalManager
from rosa_repro.global_audit import evaluate,seal_tree
from rosa_repro.dynamic_runner import run
from rosa_repro.simulator_process import ChangingDummyEpisode
from rosa_repro.execution_plan import runtime_source_manifest
from test_global_control import spec,planner,action
from rosa_repro.dummy import observation

class Sim:
 def __init__(self):self.env=ChangingDummyEpisode(10000)
 async def start(self):return dict(kind='cpu_dummy',source_manifest=runtime_source_manifest())
 async def reset(self,state,seed):
  obs,h=self.env.reset(state,seed);return dict(observation=obs,restored_state_sha256=h)
 async def step(self,a,n):return self.env.step(a,n)
 async def close(self):pass
 def abort(self):pass


async def trial(path,pipeline='P4',manager=None):
 s=spec((pipeline,));m=manager or LocalManager();c=Supervisor(s,m,planner,path/'control');await c.start()
 async def join():return await c.join('c0','arriving')
 async def leave():return await c.leave('c0-r0')
 sims={r:Sim() for r in ('c0-r0','arriving')};initial={r:dict(class_id='c0',state=np.array([1.]),seed=42) for r in sims}
 await run(c,sims,initial,[(.15,join),(.35,leave)],warmup_s=.12,duration_s=.6,output=path/'simulation')
 await c.close();return evaluate(path/'control')


@pytest.mark.parametrize('pipeline',['P1','P2','P3','P4'])
def test_live_dynamic_pipeline_recomputed_across_epochs(tmp_path,pipeline):
 value=asyncio.run(trial(tmp_path,pipeline));assert value['status']=='complete'
 assert value['metrics']['qualified_action_chunks']>0 and len(value['control_generations'])==3
 assert value['downtime_episodes'] and value['metrics']['duration_s']==.6
 assert {r['robot_id'] for r in value['audit']['records']}=={'c0-r0','arriving'}
 if pipeline=='P4':
  s1=[r for r in value['audit']['records'] if r['record_type']=='request' and r['component_id']=='s1' and r['status']=='ok'];assert all(r['plan_request_id'] for r in s1)
 assert not value['paper_metric_equivalence_claim']


def test_cold_migration_slots_remain_missing(tmp_path):
 class Slow(LocalManager):
  async def deploy(self,s):await asyncio.sleep(.1);return await super().deploy(s)
 value=asyncio.run(trial(tmp_path,'P4',Slow()));assert value['status']=='complete'
 gaps={e['episode_id'] for e in value['downtime_episodes']}
 assert gaps and any(r['status']=='missing' for r in value['metrics']['slots'] if r['episode_id'] in gaps)


def test_unknown_action_does_not_turn_into_zero_throughput(tmp_path):
 async def go():
  c=Supervisor(spec(timeout=.02),LocalManager(),planner,tmp_path/'control');await c.start();start,end=c.set_window(.01,.06);await asyncio.sleep(.015)
  v=await c.predict('c0-r0','s1',observation(1))
  async def slow(*args):await asyncio.sleep(.12);return 1
  assert (await c.apply(v['ticket_id'],slow))['status']=='unknown'
  await asyncio.sleep(.15);await c.close();return evaluate(tmp_path/'control')
 r=asyncio.run(go());assert r['status']=='incomplete' and r['metrics'] is None and r['unknown_action_generations']


def test_periodic_slot_or_s2_reference_tamper_rejected(tmp_path):
 r=asyncio.run(trial(tmp_path));folder=tmp_path/'control';(folder/'tree-evidence.json').unlink()
 gen=folder/r['control_generations'][0]['directory'];events=[json.loads(x) for x in (gen/'events.jsonl').read_text().splitlines()]
 next(e for e in events if e['kind']=='prediction_started' and e['metadata']['component_id']=='monitor')['slot_index']+=1
 (gen/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
 from rosa_repro.profile_runner import seal
 (gen/'evidence.json').unlink();seal(gen);seal_tree(folder)
 with pytest.raises(ValueError):evaluate(folder)


def test_physical_trace_rejoins_observations_plans_and_actions(tmp_path):
 from rosa_repro.dynamic_evidence import reaudit
 asyncio.run(trial(tmp_path));seal_tree(tmp_path);checked=reaudit(tmp_path)
 assert checked['status']=='complete' and checked['observations_revalidated']>0 and checked['actions_revalidated']>0
 (tmp_path/'tree-evidence.json').unlink();f=tmp_path/'simulation/request_observations.jsonl'
 rows=[json.loads(x) for x in f.read_text().splitlines()];rows[0]['context']['execution_history']='forged'
 f.write_text('\n'.join(json.dumps(x) for x in rows)+'\n');seal_tree(tmp_path)
 with pytest.raises(ValueError):reaudit(tmp_path)
