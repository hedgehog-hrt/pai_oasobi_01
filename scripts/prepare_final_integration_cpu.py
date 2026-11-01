"""Create explicitly synthetic integration fixtures; requires local OR-Tools only."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_paper_baselines import reference
from hetero_helpers import task,make_fleet
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler
from rosa_repro.execution_plan import compile_execution_plan
from rosa_repro.dynamic_control import prepare
from rosa_repro.regression import write_json
from rosa_repro.dedicated_baselines import prepare as dedicated

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 write_json(a.output/'reference.json',reference(gpus=2))
 data=make_fleet([task('c','P1',robots=2)],gpus=2,capacity=40);data['classes'][0]['action_time_s']=.005
 data['profiles'][0]['points']=[dict(rate_hz=0,mean_ms=1,p99_ms=2),dict(rate_hz=40,mean_ms=1,p99_ms=2)]
 sch=HeterogeneousScheduler(Fleet.from_dict(data)).check_rates({'c':30.});assert sch['status']=='feasible' and len(sch['placements'])==2
 plan=compile_execution_plan(data,sch,{'s1':'s1'});r0,r1=[r['robot_id'] for r in plan['classes'][0]['robots']];wid=plan['classes'][0]['robots'][0]['routes']['s1']
 spec=prepare(plan,[r0],{wid:'standby-s1'},physical_gpu_budget=3,timeout_s=3)
 write_json(a.output/'control-spec.json',spec)
 steps=[dict(op='join',robot=r1,expect='admitted'),dict(op='predict',robot=r0,component='s1',**{'as':'before'},expect='ok'),dict(op='apply',ticket='before',expect='completed'),
  dict(op='predict',robot=r1,component='s1',**{'as':'unaffected'},expect='ok'),dict(op='fault',worker=wid,expect='injected'),dict(op='health'),dict(op='failover',worker=wid,expect='ready'),
  dict(op='apply',ticket='unaffected',expect='completed'),dict(op='predict',robot=r0,component='s1',**{'as':'after'},expect='ok'),dict(op='apply',ticket='after',expect='completed'),
  dict(op='apply',ticket='after',expect='rejected_duplicate_or_unknown'),dict(op='wait',seconds=.04),dict(op='predict',robot=r0,component='s1',**{'as':'old'},expect='ok'),
  dict(op='leave',robot=r0,expect='left'),dict(op='join',robot=r0,expect='admitted'),dict(op='apply',ticket='old',expect='rejected_stale')]
 write_json(a.output/'control-scenario.json',steps)
 p4=reference(('P4',),gpus=8);profiles={w['component_id']:w['profile_id'] for w in p4['workers']}
 for method in ('gpu_per_robot','gpu_per_model_per_robot'):write_json(a.output/(method+'.json'),dedicated(p4,method,profiles,gpu_budget=8))
 print('CPU reference, R10 fault scenario and dedicated resource plans prepared')
if __name__=='__main__':main()
