import asyncio,json
from copy import deepcopy
import numpy as np
import pytest
from rosa_repro.paper_conditions import generate
from rosa_repro.paper_baselines import compile_baseline
from rosa_repro.paper_comparison import report,family
from rosa_repro.execution_plan import compile_execution_plan,validate_execution_plan,DUMMY_FACTORY
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler
from rosa_repro.local_fleet import LocalFleetTransport
from rosa_repro.closed_loop_protocol import protocol,prepare_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.paired import save_state_bank
from rosa_repro.regression import write_json
from hetero_helpers import task,make_fleet
from test_closed_loop import Sim


def reference(pipelines=('P1',),gpus=8):
 data=make_fleet([task('c'+str(i),p,robots=8//len(pipelines)) for i,p in enumerate(pipelines)],gpus=gpus,capacity=10000)
 schedule=HeterogeneousScheduler(Fleet.from_dict(data)).check_rates({c['id']:2. for c in data['classes']});assert schedule['status']=='feasible'
 return compile_execution_plan(data,schedule,{m['id']:m['id'] for c in data['classes'] for m in c['components']})


def cell(paper,kind='full_baseline',method='equal',workload='P1',rep=0):
 return next(c for c in paper['cells'] if c['experiment'] in ('R08','R09') and c['workload']==workload and sum(x['robots'] for x in c['classes'])==8 and c['comparison']==kind and c.get('method')==method and c['repeat']==rep)


def plan(ref,paper,**kw):
 c=cell(paper,**kw);profiles={w['component_id']:w['profile_id'] for w in ref['workers']} if c['comparison']=='full_baseline' else None
 return compile_baseline(ref,paper,c['id'],profiles)


def test_table3_fixed_allocation_preserves_physical_budget_and_uncapped():
 ref=reference(('P4',));paper=generate()
 a=plan(ref,paper,workload='P4');b=plan(ref,paper,workload='P4',method='weighted')
 from collections import Counter
 assert Counter(w['component_id'] for w in a['workers'])=={'s1':2,'s2':2,'safety':2,'monitor':2}
 assert Counter(w['component_id'] for w in b['workers'])=={'s1':1,'s2':3,'safety':1,'monitor':3}
 assert all(c['frequency_hz'] is None for c in b['classes']) and not b['slo_feasibility_claim']
 assert validate_execution_plan(b)==b
 b['classes'][0]['frequency_hz']=1e9
 with pytest.raises(ValueError):validate_execution_plan(b)


def test_mix2_placement_holds_reference_rates_and_batch_routes():
 ref=reference(('P2','P4'));p=plan(ref,generate(),kind='placement_only',workload='Mix2')
 assert [c['frequency_hz'] for c in p['classes']]==[c['frequency_hz'] for c in ref['classes']]
 rw={w['worker_id']:w for w in ref['workers']};bw={w['worker_id']:w for w in p['workers']}
 for a,b in zip(ref['classes'],p['classes']):
  for x,y in zip(a['robots'],b['robots']):
   for mid in x['routes']:assert rw[x['routes'][mid]]['profile_id']==bw[y['routes'][mid]]['profile_id']


@pytest.mark.parametrize('bad',['robots','budget','batch','model'])
def test_unbound_baseline_conditions_rejected(bad):
 ref=reference();paper=generate();c=cell(paper);profiles={'s1':'s1-b1'}
 if bad=='robots':c=next(x for x in paper['cells'] if x['experiment']=='R08' and x['workload']=='P1' and x['classes'][0]['robots']==16 and x['comparison']=='full_baseline')
 elif bad=='budget':paper=generate(gpu_count=4);c=cell(paper)
 elif bad=='batch':profiles={'s1':'missing-b4'}
 else:profiles={'wrong':'s1-b1'}
 with pytest.raises(ValueError):compile_baseline(ref,paper,c['id'],profiles)


async def execute(p,root,out,rep=0):
 pro=protocol(p,'cpu_smoke',warmup_s=.05,duration_s=.65,request_timeout_s=1,sim_timeout_s=3,drain_timeout_s=2)
 s=prepare_suite(p,{c['audit']['id']:'bank' for c in p['classes']},root,pro,repetitions=3,setup_budget_s=20)
 t=LocalFleetTransport(p);sims={r['robot_id']:Sim() for c in p['classes'] for r in c['robots']}
 try:return await run_closed_loop(s,root,t,sims,out,rep=rep)
 finally:await t.drain()


def test_full_prospective_cpu_family_raw_audit_ratios_and_missing(tmp_path):
 ref=reference();paper=generate();save_state_bank(tmp_path/'bank',[np.array([1.]),np.array([2.])],task='CPU',source={'kind':'synthetic'})
 write_json(tmp_path/'reference.json',ref)
 cells=[c for c in paper['cells'] if c['experiment']=='R08' and c['workload']=='P1' and c['classes'][0]['robots']==8]
 paths={c['id']:c['id'] for c in cells}
 for c in cells:
  p=ref if c['comparison']=='ROSA' else compile_baseline(ref,paper,c['id'],{'s1':'s1-b1'} if c['comparison']=='full_baseline' else None)
  r=asyncio.run(execute(p,tmp_path,tmp_path/c['id'],c['repeat']));assert r['status']=='complete',r
  assert reaudit_closed(tmp_path/c['id'])['actions_revalidated']>0
 value=report(paper,tmp_path,paths,{'R08:P1:8':'reference.json'})
 assert value['status']=='complete' and len(value['rows'])==15 and not value['gpu_results']
 assert all(v['summary']['n']==3 for v in value['groups'][0]['speedups'].values())
 paths[cells[0]['id']]='missing'
 value=report(paper,tmp_path,paths,{'R08:P1:8':'reference.json'})
 assert value['status']=='incomplete' and value['groups'][0]['speedups'] is None
 with pytest.raises(ValueError):report(paper,tmp_path,{cells[0]['id']:'missing'},{'R08:P1:8':'reference.json'})
