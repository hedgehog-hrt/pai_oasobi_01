import asyncio
from copy import deepcopy
import numpy as np
import pytest
from test_paper_baselines import reference,cell
from test_closed_loop import Sim
from rosa_repro.paper_conditions import generate
from rosa_repro.paper_baselines import compile_baseline
from rosa_repro.dedicated_comparison import prepare_family,accounting
from rosa_repro.paper_comparison import report
from rosa_repro.dedicated_core import GroupCore
from rosa_repro.dedicated_execution import launch_gate
from rosa_repro.closed_loop_runtime import run_closed_loop
from rosa_repro.local_fleet import LocalFleetTransport
from rosa_repro.paired import save_state_bank
from rosa_repro.regression import write_json

BUDGETS=dict(gpu_per_robot=8,gpu_per_model_per_robot=32)


class SyntheticGroupInventory:
    """Unit harness with explicitly synthetic process identities, never a Ray proof."""
    def __init__(self,p):
        self.groups={g['group_id']:GroupCore([w for w in p['workers'] if w['worker_id'] in g['worker_ids']],p['plan_id']) for g in p['physical_groups']}
        self.routes={w['worker_id']:self.groups[w['physical_group_id']] for w in p['workers']};self.launch_gate=launch_gate(p)
    async def reset(self,w,*a):return self.routes[w].reset(w,*a)
    async def predict(self,w,*a):return await self.routes[w].predict(w,*a)
    async def inventory(self):return {w:dict(g.stats(w),node_id='SYNTHETIC_UNIT_FIXTURE',pid=100+i,gpu_ids=[]) for i,g in enumerate(self.groups.values()) for w in g.cores}
    async def drain(self):
        for g in self.groups.values():await g.drain()


def test_paper_counts_budget_shortfall_and_no_subset():
    p=generate(dedicated_budgets=BUDGETS);ref=reference(('P4',));profiles={w['component_id']:w['profile_id'] for w in ref['workers']}
    for method,count in [('gpu_per_robot',8),('gpu_per_model_per_robot',32)]:
        c=cell(p,'dedicated_baseline',method,'P4');plan=compile_baseline(ref,p,c['id'],profiles)
        assert len(plan['physical_groups'])==count and len(plan['workers'])==32
        assert all(c['frequency_hz'] is None for c in plan['classes'])
    low=generate(dedicated_budgets={'gpu_per_model_per_robot':8});c=cell(low,'dedicated_baseline','gpu_per_model_per_robot','P4')
    resource=compile_baseline(ref,low,c['id'],profiles)
    assert resource['shortfall']==24 and resource['status']=='insufficient_resources' and len(resource['all_robots'])==8
    with pytest.raises(ValueError):generate(dedicated_budgets={'unknown':8})
    with pytest.raises(ValueError):generate(dedicated_budgets={'gpu_per_robot':True})


def family_fixture(tmp_path,budgets=BUDGETS):
    ref=reference();paper=generate(dedicated_budgets=budgets)
    save_state_bank(tmp_path/'bank',[np.array([1.]),np.array([2.]),np.array([3.])],task='CPU',source={'kind':'synthetic'})
    prepared=prepare_family(ref,paper,'R08:P1:8',{'s1':'s1-b1'},{'c0':'bank'},tmp_path,phase='cpu_smoke',
        controls=dict(warmup_s=.05,duration_s=.55,request_timeout_s=2,sim_timeout_s=3,drain_timeout_s=2),setup_budget_s=10)
    write_json(tmp_path/'reference.json',ref);return prepared


def test_entire_prepared_family_raw_reports_resources_and_failures(tmp_path):
    f=family_fixture(tmp_path);dirs={}
    async def run():
        for row in f['rows']:
            c=row['cell'];p=row['plan'];cid=c['id'];dirs[cid]=cid
            t=SyntheticGroupInventory(p) if p.get('purpose')=='dedicated_GPU_execution' else LocalFleetTransport(p)
            try:
                sims={r['robot_id']:Sim() for k in p['classes'] for r in k['robots']}
                result=await run_closed_loop(row['suite'],tmp_path,t,sims,tmp_path/cid,rep=c['repeat']);assert result['status']=='complete',result
            finally:await t.drain()
    asyncio.run(run());r=report(f['paper'],tmp_path,dirs,{'R08:P1:8':'reference.json'})
    assert r['status']=='complete' and len(r['rows'])==21 and len(r['groups'][0]['speedups'])==6
    assert all(x['resources']['actual_GPU_leases']==0 and x['qualified_Q_per_declared_GPU'] is None for x in r['rows'])
    assert all(x is None for x in r['groups'][0]['GPU_efficiency'].values())
    d=next(x for x in r['rows'] if x['cell']['method']=='gpu_per_model_per_robot')
    assert d['resources']['declared_GPU_budget']==32 and d['resources']['target_physical_groups']==8
    dirs[d['cell']['id']]='missing';bad=report(f['paper'],tmp_path,dirs,{'R08:P1:8':'reference.json'})
    assert bad['status']=='incomplete' and bad['groups'][0]['speedups'] is None


def test_infeasible_cells_have_no_executable_suite_and_remain_in_report(tmp_path):
    f=family_fixture(tmp_path,{'gpu_per_robot':1});rows=[r for r in f['rows'] if r['cell']['comparison']=='dedicated_baseline']
    assert len(rows)==3 and all(r['status']=='resource_infeasible' and r['suite'] is None for r in rows)
    r=report(f['paper'],tmp_path,{r['cell']['id']:'missing/'+r['cell']['id'] for r in f['rows']},{'R08:P1:8':'reference.json'})
    assert r['status']=='incomplete' and r['groups'][0]['status_counts']['resource_infeasible']==3


def test_dry_family_export_generates_specific_bounded_repetitions(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from paper_family import export
    f=family_fixture(tmp_path);r=export(f,tmp_path/'family',tmp_path)
    from rosa_repro.profile_adoption import read
    jobs=read(tmp_path/'family/jobs.json');assert len(jobs)==21 and not r['models_started']
    assert all('--execute' not in j['argv'] and '--case-id' in j['argv'] and '--budget-seconds' in j['argv'] for j in jobs)
    assert len(set(read(tmp_path/'family/directories.json').values()))==21
