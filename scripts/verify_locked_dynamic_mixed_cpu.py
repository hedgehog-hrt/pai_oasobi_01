"""Bounded actual CPU Ray verification; no GPU, pretrained models, or Pod access."""
import argparse,asyncio,json,os,secrets,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json
from rosa_repro.execution_plan import runtime_source_manifest,check_runtime_inventory
from rosa_repro.dynamic_protocol import inspect_trial,report as dynamic_report
from rosa_repro.dynamic_evidence import reaudit
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
 out=a.output;out.mkdir(parents=True,exist_ok=False);source=runtime_source_manifest();dirs={};dynamic=[];started=time.monotonic()
 def budget():
  if time.monotonic()-started>a.budget_seconds:raise TimeoutError('Explicit CPU proof wall budget exhausted')
 for rep in range(3):
  budget();folder=out/f'dynamic-{rep}';args=SimpleNamespace(scenario=a.fixtures/'scenario.json',root=a.fixtures,output=folder,address=None,sim_repo=None,
   solver_python=a.solver_python,sim_python=sys.executable,adoptions=None,rep=rep)
  assert await trial(args)==0
  checked=inspect_trial(folder);write_json(out/f'dynamic-{rep}-acceptance.json',checked);assert checked['status']=='accepted',checked
  raw=reaudit(folder);assert raw['observations_revalidated']>0 and raw['actions_revalidated']>0
  assert len(checked['change_gate']['recoveries'])==1 and all(r['accepted'] for r in checked['change_gate']['recoveries'])
  assert len(raw['dynamic_evaluation']['control_generations'])==5
  dirs[str(rep)]=folder.name;dynamic.append(dict(rep=rep,observations=raw['observations_revalidated'],actions=raw['actions_revalidated'],generations=5))
  print(json.dumps(dict(kind='dynamic',rep=rep,status='accepted',changes=['join','budget','fault/recovery','leave'])),flush=True)
 summary=dynamic_report(read(a.fixtures/'scenario.json'),out,dirs);assert summary['status']=='complete';write_json(out/'dynamic-report.json',summary)
 import ray
 from ray import serve
 from rosa_repro.fleet_service import deploy_fleet
 os.environ['RAY_USAGE_STATS_ENABLED']='0';os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
 counts=[]
 for name,model_count in [('p4',32),('mix2',24)]:
  budget();family_out=out/name;family_out.mkdir();f=read(a.fixtures/(name+'-family.json'));write_json(family_out/'reference.json',f['reference']);paths={}
  order=f['paper']['randomized_order'];rows=sorted(f['rows'],key=lambda r:order.index(r['cell']['id']))
  ray.init(address='local',num_cpus=model_count+2,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-mixed-',dir='/tmp'),log_to_driver=False)
  try:
   serve.start(proxy_location='Disabled')
   for row in rows:
    budget();p=row['plan'];c=row['cell'];folder=family_out/c['id']
    print(json.dumps(dict(kind='deploying',family=name,cell_id=c['id'],method=c['method'],comparison=c['comparison'])),flush=True)
    t=deploy_fleet(p)
    try:
     inv=await t.inventory();checked=check_runtime_inventory(p,inv);assert checked['unique_gpu_leases']==0
     process_count=len({(v['node_id'],v['pid']) for v in inv.values()})
     expected_groups=len(p['physical_groups']) if c['comparison']=='dedicated_baseline' else len(p['workers'])
     assert process_count==expected_groups
     if c['comparison']=='dedicated_baseline':
      assert len(inv)==model_count and process_count==(8 if c['method']=='gpu_per_robot' else model_count)
     sims={r['robot_id']:Sim() for cl in p['classes'] for r in cl['robots']}
     result=await run_closed_loop(row['suite'],a.fixtures,t,sims,folder,rep=c['repeat']);assert result['status']=='complete',result
     audit=reaudit_closed(folder);assert audit['actions_revalidated']>0
     requests=[json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
     assert any(x['plan_request_id'] for x in requests if x['component_id']=='s1')
     assert {'s1','s2','safety','monitor'}<={x['component_id'] for x in requests if x['status']=='ok'}
     counts.append(dict(family=name,cell_id=c['id'],comparison=c['comparison'],method=c['method'],rep=c['repeat'],actions=audit['actions_revalidated'],
      actual_actor_processes=process_count,model_workers=len(inv),robots=8,actual_GPU_leases=0))
     paths[c['id']]=folder.name
    finally:t.close()
    print(json.dumps(dict(kind='comparison',family=name,cell_id=c['id'],status='complete',actors=process_count)),flush=True)
  finally:serve.shutdown();ray.shutdown()
  r=paper_report(f['paper'],family_out,paths,{f['family']:'reference.json'});write_json(family_out/'report.json',r);write_json(family_out/'directories.json',paths)
  assert r['status']=='complete' and len(r['rows'])==21 and not r['gpu_results'],r
  missing=dict(paths);missing[next(iter(missing))]='missing'
  bad=paper_report(f['paper'],family_out,missing,{f['family']:'reference.json'});assert bad['status']=='incomplete' and bad['groups'][0]['speedups'] is None
  write_json(family_out/'missing-cell-report.json',bad)
 assert runtime_source_manifest()==source
 write_json(out/'verification.json',dict(status='passed',source_manifest=source,actual_CPU_Ray_trials=45,dynamic=dynamic,comparison_cells=counts,
  evaluated_dynamic_repetitions=3,paper_comparison_trials=42,dynamic_simulator_process_starts=12,
  comparison_simulators='eight in-process changing dummy episodes per trial',own_clusters_stopped=True,
  pretrained_models_loaded=False,GPU_execution=False,pod_access_attempted=False,paper_performance_claim=False,
  elapsed_s=time.monotonic()-started))
 print('CPU Ray dynamic recovery and full P4/Mix2 comparison verified; no GPU performance claim.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-python',required=True)
 p.add_argument('--execute',action='store_true');p.add_argument('--budget-seconds',type=float,default=3600);a=p.parse_args()
 if not 600<=a.budget_seconds<=7200:raise ValueError('600..7200 seconds required')
 if not a.execute:print('Planned: 3 dynamic + 42 comparison trials; no processes started.')
 else:asyncio.run(proof(a))
