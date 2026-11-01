import asyncio
import numpy as np
import pytest
from test_dedicated_comparison import SyntheticGroupInventory
from test_paper_baselines import reference
from test_closed_loop import Sim
from rosa_repro.paper_conditions import generate
from rosa_repro.dedicated_comparison import prepare_family
from rosa_repro.paper_comparison import report
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.local_fleet import LocalFleetTransport
from rosa_repro.paired import save_state_bank
from rosa_repro.regression import write_json


@pytest.mark.parametrize('pipelines,key,models',[(('P4',),'R08:P4:8',32),(('P2','P4'),'R09:Mix2:8',24)])
def test_full_p4_and_mixed_families_preserve_all_variants_states_and_groups(tmp_path,pipelines,key,models):
 ref=reference(pipelines);paper=generate(dedicated_budgets=dict(gpu_per_robot=8,gpu_per_model_per_robot=models))
 save_state_bank(tmp_path/'bank',[np.array([1.]),np.array([2.]),np.array([3.])],task='CPU',source={'kind':'synthetic'})
 f=prepare_family(ref,paper,key,{w['component_id']:w['profile_id'] for w in ref['workers']},{c['audit']['id']:'bank' for c in ref['classes']},tmp_path,
  phase='cpu_smoke',controls=dict(warmup_s=.1,duration_s=2.8,request_timeout_s=2,sim_timeout_s=3,drain_timeout_s=2),setup_budget_s=30)
 write_json(tmp_path/'reference.json',ref);dirs={}
 async def go():
  for row in f['rows']:
   c=row['cell'];p=row['plan'];folder=tmp_path/c['id'];dirs[c['id']]=folder.name
   if c['comparison']=='dedicated_baseline':
    assert len(p['workers'])==models
    assert len(p['physical_groups'])==(8 if c['method']=='gpu_per_robot' else models)
   t=SyntheticGroupInventory(p) if c['comparison']=='dedicated_baseline' else LocalFleetTransport(p)
   try:
    r=await run_closed_loop(row['suite'],tmp_path,t,{r['robot_id']:Sim() for cl in p['classes'] for r in cl['robots']},folder,rep=c['repeat'])
    assert r['status']=='complete',r
    assert reaudit_closed(folder)['actions_revalidated']>0
   finally:await t.drain()
 asyncio.run(go());r=report(paper,tmp_path,dirs,{key:'reference.json'})
 assert r['status']=='complete' and len(r['rows'])==21
 assert len(r['groups'][0]['speedups'])==6
 for x in r['groups'][0]['speedups'].values():
  assert len(x['pairs'])==3
  if any(pair['ratio'] is None for pair in x['pairs']):assert x['summary'] is None and not x['complete']
  else:assert x['summary']['n']==3 and x['complete']
 assert not r['gpu_results'] and all(x['qualified_Q_per_declared_GPU'] is None for x in r['rows'])
 # A repeated run cannot substitute for the missing prospective repetition.
 ids=list(dirs);dirs[ids[1]]=dirs[ids[0]]
 with pytest.raises(ValueError,match='Distinct'):report(paper,tmp_path,dirs,{key:'reference.json'})


def test_export_dry_then_execute_have_distinct_reports_and_overhead_budget(tmp_path):
 import sys
 from pathlib import Path
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from paper_family import export
 from test_dedicated_comparison import family_fixture
 from rosa_repro.profile_adoption import read
 f=family_fixture(tmp_path);export(f,tmp_path/'family',tmp_path)
 jobs=read(tmp_path/'family/jobs.json');byid={r['cell']['id']:r for r in f['rows']}
 for j in jobs:
  dry=j['argv'];run=j['execute_argv']
  assert '--execute' not in dry and run[-1]=='--execute'
  assert dry[dry.index('--report')+1]!=run[run.index('--report')+1]
  assert float(run[run.index('--budget-seconds')+1])>byid[j['cell_id']]['suite']['trial_wall_budget_s']+10
