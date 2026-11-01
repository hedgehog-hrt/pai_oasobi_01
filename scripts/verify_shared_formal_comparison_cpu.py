"""Actual local CPU Ray proof, with fixed prospective repetitions and raw re-audit."""
import argparse,asyncio,json,os,secrets,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json
from rosa_repro.execution_plan import runtime_source_manifest,check_runtime_inventory
from rosa_repro.dynamic_protocol import inspect_trial,report as dynamic_report
from rosa_repro.paper_comparison import report as paper_report
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.simulator_process import ChangingDummyEpisode

class Sim:
    def __init__(self):self.env=ChangingDummyEpisode()
    async def start(self):return dict(kind='cpu_dummy',source_manifest=runtime_source_manifest())
    async def reset(self,state,seed):
        obs,h=self.env.reset(state,seed);return dict(observation=obs,restored_state_sha256=h,success=False)
    async def step(self,action,n):return self.env.step(action,n)
    async def close(self):pass

async def proof(a):
    from global_experiments import trial
    out=a.output;out.mkdir(parents=True,exist_ok=False);source=runtime_source_manifest();dirs={};dynamic=[]
    for rep in range(3):
        folder=out/f'dynamic-{rep}';args=SimpleNamespace(scenario=a.fixtures/'scenario.json',root=a.fixtures,output=folder,address=None,sim_repo=None,
            solver_python=a.solver_python,sim_python=sys.executable,adoptions=None,rep=rep)
        assert await trial(args)==0
        checked=inspect_trial(folder);write_json(out/f'dynamic-{rep}-acceptance.json',checked)
        assert checked['status']=='accepted',checked
        dirs[str(rep)]=folder.name;dynamic.append(checked)
        print(json.dumps(dict(kind='dynamic',rep=rep,status='accepted')),flush=True)
    summary=dynamic_report(read(a.fixtures/'scenario.json'),out,dirs);assert summary['status']=='complete';write_json(out/'dynamic-report.json',summary)
    args.scenario=a.fixtures/'warmup-failure.json';args.output=out/'warmup-failure';args.rep=0
    await trial(args);negative=inspect_trial(args.output);assert negative['status']=='not_accepted' and not negative['warmup_gate']['requirements_met']
    assert not list((args.output/'simulation').glob('action-*.npz'));write_json(out/'warmup-failure-acceptance.json',negative)
    print(json.dumps(dict(kind='warmup_negative',status='passed',physical_actions=0)),flush=True)
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_fleet
    os.environ['RAY_USAGE_STATS_ENABLED']='0';os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    ray.init(address='local',num_cpus=10,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-compare-',dir='/tmp'),log_to_driver=False)
    f=read(a.fixtures/'family.json');write_json(out/'reference.json',f['reference']);paths={};counts=[]
    order=f['paper']['randomized_order'];rows=sorted(f['rows'],key=lambda r:order.index(r['cell']['id']))
    try:
        serve.start(proxy_location='Disabled')
        for row in rows:
            p=row['plan'];c=row['cell'];folder=out/c['id'];t=deploy_fleet(p)
            try:
                inv=await t.inventory();checked=check_runtime_inventory(p,inv);assert checked['unique_gpu_leases']==0
                sims={r['robot_id']:Sim() for cl in p['classes'] for r in cl['robots']}
                result=await run_closed_loop(row['suite'],a.fixtures,t,sims,folder,rep=c['repeat']);assert result['status']=='complete',result
                audit=reaudit_closed(folder);assert audit['actions_revalidated']>0
                counts.append(dict(cell_id=c['id'],method=c['method'],rep=c['repeat'],actions=audit['actions_revalidated'],
                    actual_actor_processes=len({(v['node_id'],v['pid']) for v in inv.values()}),actual_GPU_leases=0))
                paths[c['id']]=folder.name
            finally:t.close()
            print(json.dumps(dict(kind='paper_comparison',cell_id=c['id'],status='complete')),flush=True)
    finally:serve.shutdown();ray.shutdown()
    r=paper_report(f['paper'],out,paths,{'R08:P1:8':'reference.json'});write_json(out/'paper-report.json',r);write_json(out/'paper-directories.json',paths)
    assert r['status']=='complete' and len(r['rows'])==15 and not r['gpu_results'],r
    missing=dict(paths);missing[next(iter(missing))]='missing'
    bad=paper_report(f['paper'],out,missing,{'R08:P1:8':'reference.json'});assert bad['status']=='incomplete' and bad['groups'][0]['speedups'] is None
    write_json(out/'missing-cell-report.json',bad);assert runtime_source_manifest()==source
    write_json(out/'verification.json',dict(status='passed',source_manifest=source,actual_CPU_Ray_trials=19,
        evaluated_dynamic_repetitions=3,warmup_negative_checks=1,paper_comparison_trials=15,paper_cells=counts,
        dynamic_simulator_process_starts=16,paper_simulators='eight in-process changing dummy episodes per trial',
        own_clusters_stopped=True,models_loaded=False,GPU_execution=False,pod_access_attempted=False,paper_performance_claim=False,
        note='For P1 both dedicated methods have the same target grouping; P4 grouping differences are covered by local contract tests and the prior v0.17 CPU proof.'))
    print('All CPU integration checks passed; no GPU/model/Pod results claimed.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-python',required=True)
    asyncio.run(proof(p.parse_args()))
