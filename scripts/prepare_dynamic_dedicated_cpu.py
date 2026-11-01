"""Synthetic CPU fixtures only. No downloaded models or GPU measurements."""
import argparse,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from runtime_helpers import plan_for
from rosa_repro.execution_plan import compile_execution_plan
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler
from rosa_repro.global_planner import catalog_from_plan
from rosa_repro.global_control import prepare as global_spec
from rosa_repro.global_scenario import prepare as scenario
from rosa_repro.dedicated_baselines import prepare as resources
from rosa_repro.dedicated_execution import compile_plan
from rosa_repro.regression import write_json
from rosa_repro.paired import save_state_bank
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
 ref=plan_for(('P2','P3','P4'));x=ref['inputs'];fleet=x['fleet']
 for c in fleet['classes']:c['action_time_s']=.05
 fleet['search']['time_limit_s']=1;fleet['search']['max_iterations']=20
 decision=HeterogeneousScheduler(Fleet.from_dict(fleet)).check_rates({c['id']:10. for c in fleet['classes']})
 ref=compile_execution_plan(fleet,decision,x['roles']);cat=catalog_from_plan(ref)
 spec=global_spec(cat,{c['id']:[c['id']+'-r0'] for c in fleet['classes']},4,call_timeout_s=10,deployment_timeout_s=60)
 save_state_bank(out/'bank',[np.array([1.]),np.array([2.]),np.array([3.])],task='CPU dynamic fixture',source={'kind':'synthetic'})
 changes=[dict(op='join',offset_s=5.,class_id='c2',robot_id='arrival'),dict(op='budget',offset_s=25.,gpu_budget=5),
  dict(op='fault',offset_s=45.,robot_id='c2-r0',component_id='monitor'),dict(op='leave',offset_s=65.,robot_id='c0-r0')]
 s=scenario(spec,{c:'bank' for c in spec['initial_population']},out,changes,warmup_s=1,duration_s=90,setup_budget_s=90,
  max_steps=100000,max_capture_bytes=1024**3,health=dict(interval_s=.2,timeout_s=3.,max_recoveries=2))
 write_json(out/'scenario.json',s);write_json(out/'global-spec.json',spec);write_json(out/'reference.json',ref)
 one=plan_for(('P4',))
 for method in ('gpu_per_robot','gpu_per_model_per_robot'):
  write_json(out/(method+'.json'),compile_plan(resources(one,method,{w['component_id']:w['profile_id'] for w in one['workers']},gpu_budget=4)))
 print('CPU dynamic and dedicated fixtures prepared; no GPU evidence')
