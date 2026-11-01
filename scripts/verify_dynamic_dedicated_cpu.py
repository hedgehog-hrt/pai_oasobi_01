"""Actual local CPU Ray proof: global replacement, live P2-P4, both dedicated executors."""
import argparse,asyncio,json,os,secrets,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json
from rosa_repro.execution_plan import runtime_source_manifest,check_runtime_inventory
from rosa_repro.dynamic_evidence import reaudit as dynamic_audit
from rosa_repro.closed_loop_protocol import protocol,prepare_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.simulator_process import ProcessSimulator
from rosa_repro.global_audit import verify_tree


async def proof(a):
 from global_experiments import trial
 out=a.output;out.mkdir(parents=True,exist_ok=False);source=runtime_source_manifest()
 args=SimpleNamespace(scenario=a.fixtures/'scenario.json',root=a.fixtures,output=out/'dynamic',address=None,sim_repo=None,
  solver_python=a.solver_python,sim_python=sys.executable,adoptions=None)
 assert await trial(args)==0
 d=dynamic_audit(out/'dynamic');write_json(out/'dynamic-reaudit.json',d);e=d['dynamic_evaluation']
 assert d['status']=='complete' and d['actions_revalidated']>0 and e['metrics']['qualified_action_chunks']>0
 journal=[json.loads(x) for x in (out/'dynamic/control/journal.jsonl').read_text().splitlines()]
 assert len(e['control_generations'])==5 and any(r['kind']=='health_recovery_result' and r['result']['status']=='committed' for r in journal)
 assert {c['pipeline'] for c in e['audit']['classes']}=={'P2','P3','P4'}
 assert all(r['result']['status'] in ('committed','injected') for r in journal if r['kind']=='scheduled_change_result')
 assert any(r['status']=='missing' for r in e['metrics']['slots']) and e['downtime_episodes']
 print(json.dumps(dict(kind='global_dynamic_P2_P4',status='passed',generations=5,actions=d['actions_revalidated'])),flush=True)
 import ray
 from ray import serve
 from rosa_repro.fleet_service import deploy_fleet
 os.environ['RAY_USAGE_STATS_ENABLED']='0';os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
 ray.init(address='local',num_cpus=6,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-dedicated-',dir='/tmp'),log_to_driver=False)
 rows=[]
 try:
  serve.start(proxy_location='Disabled')
  for method,count in (('gpu_per_robot',1),('gpu_per_model_per_robot',4)):
   plan=read(a.fixtures/(method+'.json'));folder=out/method;folder.mkdir();t=deploy_fleet(plan);sims={}
   try:
    inventory=await t.inventory();checked=check_runtime_inventory(plan,inventory);assert checked['physical_groups']==count and checked['unique_gpu_leases']==0
    p=protocol(plan,'cpu_smoke',warmup_s=1,duration_s=3,request_timeout_s=10,sim_timeout_s=10,drain_timeout_s=10)
    suite=prepare_suite(plan,{'c0':'bank'},a.fixtures,p,repetitions=1,setup_budget_s=60)
    for c in plan['classes']:
     for r in c['robots']:sims[r['robot_id']]=ProcessSimulator(dict(kind='cpu_dummy'),python=sys.executable,stderr_path=folder/f'simulator-{len(sims)}.log',timeout_s=10)
    result=await run_closed_loop(suite,a.fixtures,t,sims,folder/'raw');assert result['status']=='complete',result
    audit=reaudit_closed(folder/'raw');assert audit['actions_revalidated']>0
    after=await t.inventory();assert all(v['group_peak_invocations']==1 and not v['inflight_batches'] and not v['busy_sessions'] for v in after.values())
    rows.append(dict(method=method,raw_directory=method+'/raw',physical_actor_groups=count,logical_models=4,unique_gpu_leases=0,actions_revalidated=audit['actions_revalidated']))
   finally:
    for sim in sims.values():await sim.close()
    t.close();write_json(folder/'cleanup.json',dict(own_applications_deleted=True,simulators_closed=True))
   print(json.dumps(dict(kind=method,status='passed',physical_actor_groups=count)),flush=True)
 finally:serve.shutdown();ray.shutdown()
 assert runtime_source_manifest()==source
 result=dict(status='passed',source_manifest=source,actual_CPU_Ray_trials=3,actual_CPU_simulator_process_starts=6,global_generations=5,dedicated_trials=rows,
  dynamic_observations_revalidated=d['observations_revalidated'],dynamic_actions_revalidated=d['actions_revalidated'],dynamic_duration_s=e['metrics']['duration_s'],
  own_clusters_stopped=True,GPU_execution=False,pod_access_attempted=False,pretrained_weights_loaded=False,paper_performance_claim=False)
 write_json(out/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--solver-python',required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(proof(p.parse_args()))
