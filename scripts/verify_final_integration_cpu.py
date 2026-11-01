"""Real CPU Ray/Serve + simulator-process proof; no Pod/GPU/pretrained models."""
import argparse,asyncio,json,os,secrets,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.regression import write_json,save_observation,sha256
from rosa_repro.paired import save_state_bank,digest_value
from rosa_repro.paper_conditions import generate as paper
from rosa_repro.paper_baselines import compile_baseline
from rosa_repro.paper_comparison import report as comparison
from rosa_repro.closed_loop_protocol import protocol,prepare_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.simulator_process import ProcessSimulator
from rosa_repro.dynamic_control import Controller,reaudit as control_audit
from rosa_repro.control_scenario import run as control_run
from rosa_repro.dummy import observation
from rosa_repro.r05_protocol import template,timing,prepare as r05_prepare,plan_for
from rosa_repro.r05_runtime import run_trial,reaudit as r05_audit
from rosa_repro.r05_prediction import generate as pilot_prediction,verify as verify_prediction
from rosa_repro.r05_load_curves import prepare as curve_suite,curve_plan,generate as curve_prediction
from rosa_repro.profile_runner import run_probe,reaudit as probe_audit
from rosa_repro.profile_protocol import build_corpus,load_corpus
from rosa_repro.r05_workflow import check_pilots,report as r05_report
from rosa_repro.execution_plan import DUMMY_FACTORY
read=lambda p:json.loads(Path(p).read_text())

async def proof(out,fixtures):
 import ray
 from ray import serve
 from rosa_repro.fleet_service import deploy_fleet,deploy_control,deploy_r05,deploy_r05_curve
 source=runtime_source_manifest();out.mkdir(parents=True,exist_ok=False);trials=[];sim_starts=0
 os.environ['RAY_USAGE_STATS_ENABLED']='0';os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
 ray.init(address='local',num_cpus=5,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-final-',dir='/tmp'),log_to_driver=False)
 try:
  serve.start(proxy_location='Disabled');ref=read(fixtures/'reference.json');book=paper(gpu_count=2);base=out/'paper';base.mkdir()
  write_json(base/'reference.json',ref);write_json(base/'paper.json',book);save_state_bank(base/'bank',[np.array([1.]),np.array([2.])],task='CPU',source={'kind':'synthetic'})
  cells=[c for c in book['cells'] if c['experiment']=='R08' and c['workload']=='P1' and c['classes'][0]['robots']==8];paths={}
  for cell in cells:
   p=ref if cell['comparison']=='ROSA' else compile_baseline(ref,book,cell['id'],{'s1':'s1-b1'} if cell['comparison']=='full_baseline' else None)
   pro=protocol(p,'cpu_smoke',warmup_s=.5,duration_s=1.2,request_timeout_s=10,sim_timeout_s=10,drain_timeout_s=15)
   suite=prepare_suite(p,{'c0':'bank'},base,pro,repetitions=3,setup_budget_s=60);t=deploy_fleet(p);sims={};directory=base/cell['id'];directory.mkdir()
   try:
    for cls in p['classes']:
     for r in cls['robots']:
      sims[r['robot_id']]=ProcessSimulator(dict(kind='cpu_dummy'),python=sys.executable,stderr_path=directory/f'sim-{len(sims)}.log',timeout_s=10);sim_starts+=1
    result=await run_closed_loop(suite,base,t,sims,directory/'raw',rep=cell['repeat']);assert result['status']=='complete',result
    audit=reaudit_closed(directory/'raw');assert audit['actions_revalidated']>0;paths[cell['id']]=cell['id']+'/raw';trials.append(dict(kind='paper',directory=str((directory/'raw').relative_to(out))))
   finally:
    for sim in sims.values():await sim.close()
    t.close();write_json(directory/'cleanup.json',dict(own_deployments_removed=True,simulators_closed=True))
   print(json.dumps(dict(kind='paper',cell=cell['id'],status='passed')),flush=True)
  refs={'R08:P1:8':'reference.json'};r=comparison(book,base,paths,refs);assert r['status']=='complete'
  for name,val in [('directories.json',paths),('references.json',refs),('report.json',r),('figure-request.json',dict(kind='paper_comparison',paper=book,directories=paths,references=refs))]:write_json(base/name,val)
  missing=dict(paths);missing[cells[0]['id']]='not-executed';bad=comparison(book,base,missing,refs);assert bad['groups'][0]['speedups'] is None;write_json(base/'missing-report.json',bad)
  spec=read(fixtures/'control-spec.json');steps=read(fixtures/'control-scenario.json');t=deploy_control(spec);c=Controller(spec,t,out/'control')
  try:
   await c.start();await control_run(c,t,steps);r=await c.close();assert r['status']=='complete' and r['actions_completed']==3
   assert control_audit(out/'control')==r;trials.append(dict(kind='control',directory='control'))
  finally:
   if not c.closed:await c.close()
   t.close()
  print(json.dumps(dict(kind='control',status='passed')),flush=True)
  base=out/'curves';base.mkdir();corpus_root=base/'observations'
  for i in range(4):save_observation(corpus_root/str(i),observation(i+1),dict(task='synthetic-R05',case={'state_sha256':digest_value([i])},execution_kind='cpu_dummy'))
  corpus=build_corpus(corpus_root,kind='synthetic_cpu');cfg=template();cfg.update(mode='cpu_dummy',hardware_kind='cpu-fixture',model_revision='synthetic',runtime_fingerprint='curve-CPU',binding=dict(factory=DUMMY_FACTORY,kwargs=dict(delay_s=.005)),batch_wait_ms=10,watchdog_s=10,drain_s=10,setup_budget_s=60)
  pro=timing('cpu_smoke',warmup_s=.4,duration_s=.6);pilots=r05_prepare(paper(),cfg,corpus,stage='pilot',groups=['k4-a50'],protocol=pro);deps={'k4-a50':[]}
  for cell in pilots['cells']:
   p=plan_for(cfg,cell);t=deploy_r05(p);directory=base/('pilot-'+str(cell['batch_size']))
   try:r=await run_trial(pilots,corpus_root,cell['id'],t,directory);assert r['status']=='complete',r;r05_audit(directory)
   finally:t.close()
   deps['k4-a50'].append(directory.name);trials.append(dict(kind='r05',directory=str(directory.relative_to(out))))
  pilot_prediction(base,deps);collection=curve_suite(pilots,[10.,20.],1);_,samples=load_corpus(corpus,corpus_root,'profile','cpu_dummy');profiles={'k4-a50':{str(b):[] for b in (1,2,4,8,16)}}
  for case in collection['cases']:
   p=curve_plan(cfg,case['r05_cell']);t=deploy_r05_curve(p);directory=base/case['case_id']
   try:r=await run_probe(p,t,samples,case,directory,corpus_id=corpus['corpus_id'],split='profile');assert r['status']=='complete',r;probe_audit(directory)
   finally:t.close()
   profiles['k4-a50'][str(case['batch_cap'])].append(directory.name);trials.append(dict(kind='curve',directory=str(directory.relative_to(out))))
  pred=curve_prediction(base,deps,profiles);verify_prediction(pred,base);write_json(base/'prediction.json',pred);h=sha256(base/'prediction.json');text=(base/'prediction.json').read_text()
  selection={'k4-a50':dict(batch_size=pred['groups']['k4-a50'],prediction_evidence_sha256=h)}
  confirm=r05_prepare(paper(r05_selection=selection),cfg,corpus,stage='confirmation',groups=['k4-a50'],protocol=pro,prediction_artifacts={h:text});dependency=check_pilots(confirm,base,deps);paths={}
  for cell in confirm['cells']:
   t=deploy_r05(plan_for(cfg,cell));directory=base/cell['id']
   try:r=await run_trial(confirm,corpus_root,cell['id'],t,directory,pilot_runtime=dependency[cell['group']]['backend_attestation']);assert r['status']=='complete',r;r05_audit(directory)
   finally:t.close()
   paths[cell['id']]=directory.name;trials.append(dict(kind='r05',directory=str(directory.relative_to(out))))
  comparison_result=r05_report(confirm,base,paths);assert comparison_result['status']=='complete'
  for name,val in [('pilot-suite.json',pilots),('pilot-dependencies.json',deps),('curve-suite.json',collection),('profile-directories.json',profiles),('confirmation-suite.json',confirm),('confirmation-directories.json',paths),('confirmation-report.json',comparison_result)]:write_json(base/name,val)
  assert source==runtime_source_manifest();write_json(out/'verification.json',dict(status='passed',source_manifest=source,trials=trials,actual_CPU_Ray_trials=len(trials),simulator_process_starts=sim_starts,
   GPU_execution=False,pod_access_attempted=False,pretrained_weights_loaded=False,real_LIBERO_tasks_run=0,paper_equivalence_claim=False,selected_batch=pred['groups']['k4-a50']))
 finally:
  serve.shutdown();ray.shutdown();write_json(out/'cleanup.json',dict(own_cluster_stopped=True,driver_disconnected=True))
 print(json.dumps(dict(status='passed',actual_CPU_Ray_trials=len(trials))),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);a=p.parse_args();asyncio.run(proof(a.output.resolve(),a.fixtures.resolve()))
if __name__=='__main__':main()
