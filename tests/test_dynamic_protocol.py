import asyncio,json,time
from copy import deepcopy
import numpy as np
import pytest
from test_global_control import spec,planner
from test_dynamic_evaluation import Sim
from rosa_repro.global_control import Supervisor
from rosa_repro.global_managers import LocalManager
from rosa_repro.global_scenario import prepare
from rosa_repro.dynamic_runner import run
from rosa_repro.dynamic_protocol import protocol,inspect_trial,report,sample_counts
from rosa_repro.global_audit import seal_tree
from rosa_repro.paired import save_state_bank,load_state_bank
from rosa_repro.regression import write_json


def scenario(tmp_path,**kw):
 s=spec(('P4',));save_state_bank(tmp_path/'bank',[np.array([1.]),np.array([2.]),np.array([3.])],task='CPU',source={'kind':'synthetic'})
 p=protocol(s,phase='cpu_smoke',warmup_s=.2,duration_s=.5,**kw)
 return prepare(s,{'c0':'bank'},tmp_path,[dict(op='join',offset_s=.15,class_id='c0',robot_id='new')],warmup_s=.2,duration_s=.5,evaluation=p,repetitions=3)


def trial(s,root,out,rep=0):
 async def go():
  out.mkdir();write_json(out/'scenario.json',s);write_json(out/'trial_condition.json',dict(scenario_id=s['id'],rep=rep))
  c=Supervisor(s['spec'],LocalManager(),planner,out/'control');await c.start();m,states=load_state_bank(root/'bank');initial={}
  for i,(rid,cid) in enumerate(s['robots'].items()):
   case=m['cases'][(i+rep)%len(m['cases'])];initial[rid]=dict(class_id=cid,state=states[case['case_id']],seed=case['seed'])
  await run(c,{r:Sim() for r in initial},initial,[(.15,lambda:c.join('c0','new'))],warmup_s=s['warmup_s'],duration_s=s['duration_s'],output=out/'simulation',evaluation=s['evaluation'])
  await c.close();seal_tree(out);return inspect_trial(out)
 return asyncio.run(go())


def test_prospective_warmup_arrivals_and_repeated_report(tmp_path):
 s=scenario(tmp_path);dirs={}
 for rep in range(3):
  r=trial(s,tmp_path,tmp_path/str(rep),rep);assert r['status']=='accepted' and not r['formal_protocol_met'];dirs[str(rep)]=str(rep)
  assert all(m['p99_ms'] is None for c in r['latency'].values() for m in c.values())
 result=report(s,tmp_path,dirs);assert result['status']=='complete' and result['weighted_Q']['n']==3
 assert report(s,tmp_path,dict(dirs,**{'2':'missing'}))['weighted_Q'] is None


def test_warmup_failure_stops_before_physical_actions(tmp_path):
 s=scenario(tmp_path,warmup_min_success=10000);begin=time.monotonic();r=trial(s,tmp_path,tmp_path/'run')
 assert r['status']=='not_accepted' and not r['warmup_gate']['requirements_met'] and time.monotonic()-begin<2
 assert not list((tmp_path/'run/simulation').glob('action-*.npz')) and r['weighted_Q'] is None


def test_sample_and_change_failures_remain_unaccepted(tmp_path):
 s=scenario(tmp_path,measure_min_success=10000);r=trial(s,tmp_path,tmp_path/'run');assert not r['sample_gate']['requirements_met'] and r['weighted_Q'] is None
 assert r['raw_weighted_Q'] is not None


def test_late_client_return_cannot_count_as_warmup():
 s=spec();base=dict(robot_id='c0-r0',component_id='s1',status='ok',invoked_ns=1,returned_ns=9)
 r=sample_counts(s,{'c0-r0':'c0'},[base,dict(base,returned_ns=10),dict(base,status='error')],start=0,end=10,minimum=2)
 assert r['counts']=={'c0/s1':1} and not r['requirements_met']


def test_cpu_cannot_request_formal_and_repetitions_cannot_shrink(tmp_path):
 with pytest.raises(ValueError):protocol(spec(),phase='formal',warmup_s=30,duration_s=120)
 s=scenario(tmp_path)
 with pytest.raises(ValueError):prepare(s['spec'],{'c0':'bank'},tmp_path,s['changes'],warmup_s=.2,duration_s=.5,evaluation=s['evaluation'],repetitions=1)

def test_failed_simulator_cleanup_invalidates_formal_acceptance(tmp_path):
 s=scenario(tmp_path);out=tmp_path/'run';assert trial(s,tmp_path,out)['status']=='accepted'
 (out/'simulation/cleanup.json').write_text(json.dumps(dict(status='failed',errors=['simulator_cleanup:TimeoutError'])))
 (out/'tree-evidence.json').unlink()
 seal_tree(out)
 assert inspect_trial(out)['status']=='not_accepted'

def test_changed_prospective_reset_identity_is_rejected(tmp_path):
 s=scenario(tmp_path);out=tmp_path/'run';trial(s,tmp_path,out)
 p=out/'simulation/initial_states.json';value=json.loads(p.read_text());next(iter(value.values()))['seed']+=1;p.write_text(json.dumps(value));(out/'tree-evidence.json').unlink();seal_tree(out)
 with pytest.raises(ValueError,match='state/seed'):inspect_trial(out)
