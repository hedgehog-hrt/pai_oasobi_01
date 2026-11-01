"""Generate synthetic current-source fixtures for the v0.18 CPU integration proof."""
import argparse,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from runtime_helpers import plan_for
from test_paper_baselines import reference
from rosa_repro.execution_plan import compile_execution_plan
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler
from rosa_repro.global_planner import catalog_from_plan
from rosa_repro.global_control import prepare as spec
from rosa_repro.global_scenario import prepare as scenario
from rosa_repro.dynamic_protocol import protocol
from rosa_repro.paper_conditions import generate
from rosa_repro.dedicated_comparison import prepare_family
from rosa_repro.regression import write_json
from rosa_repro.paired import save_state_bank

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    save_state_bank(out/'bank',[np.array([1.]),np.array([2.]),np.array([3.])],task='CPU synthetic only',source={'kind':'synthetic'})
    ref=plan_for(('P2','P3','P4'));x=ref['inputs'];fleet=x['fleet']
    for c in fleet['classes']:c['action_time_s']=.05
    fleet['search']['time_limit_s']=1;fleet['search']['max_iterations']=20
    decision=HeterogeneousScheduler(Fleet.from_dict(fleet)).check_rates({c['id']:10. for c in fleet['classes']})
    ref=compile_execution_plan(fleet,decision,x['roles']);cat=catalog_from_plan(ref)
    s=spec(cat,{c['id']:[c['id']+'-r0'] for c in fleet['classes']},4,call_timeout_s=10,deployment_timeout_s=60)
    changes=[dict(op='join',offset_s=3.,class_id='c2',robot_id='arrival')]
    for name,minimum in [('scenario',1),('warmup-failure',10000)]:
        pro=protocol(s,phase='cpu_smoke',warmup_s=2.,duration_s=30.,warmup_min_success=minimum,change_tolerance_s=1.)
        value=scenario(s,{c:'bank' for c in s['initial_population']},out,changes,warmup_s=2.,duration_s=30.,setup_budget_s=90,
            max_steps=100000,health=dict(interval_s=.5,timeout_s=10.,max_recoveries=2),evaluation=pro,repetitions=3)
        write_json(out/(name+'.json'),value)
    ref=reference(gpus=1);paper=generate(gpu_count=1,dedicated_budgets=dict(gpu_per_robot=8,gpu_per_model_per_robot=8))
    value=prepare_family(ref,paper,'R08:P1:8',{'s1':'s1-b1'},{'c0':'bank'},out,phase='cpu_smoke',
        controls=dict(warmup_s=.5,duration_s=2.,request_timeout_s=10.,sim_timeout_s=10.,drain_timeout_s=10.),setup_budget_s=90)
    write_json(out/'family.json',value);write_json(out/'reference.json',ref)
    print('Prepared three dynamic repetitions, a warmup-failure check, and fifteen paper comparison cells; CPU only.')
