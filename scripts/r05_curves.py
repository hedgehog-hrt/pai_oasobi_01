"""Bounded measured-load collection for R05; preserves finite stream count and partial batches."""
import argparse,asyncio,json,math,os,secrets,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.r05_load_curves import prepare,validate_collection,curve_plan
from rosa_repro.r05_prediction import generate as verify_pilots
from rosa_repro.profile_protocol import load_corpus
from rosa_repro.profile_runner import run_probe,reaudit
from rosa_repro.regression import write_json
from rosa_repro.resume_workflow import supervise
read=lambda p:json.loads(Path(p).read_text())


def inputs(a):
 s=validate_collection(read(a.suite));cells=[c for c in s['cases'] if c['case_id']==a.case]
 if len(cells)!=1:raise ValueError('Select one predeclared case')
 case=cells[0];p=curve_plan(s['pilot_suite']['config'],case['r05_cell']);mode=p['mode']
 if (mode=='gpu' and (not a.address or a.address=='local')) or (mode=='cpu_dummy' and a.address):raise ValueError('GPU needs existing address; CPU refuses external clusters')
 pred=verify_pilots(a.evidence_root,read(a.pilots));prior=s['pilot_suite']
 from rosa_repro.execution_plan import digest
 if pred['config_sha256']!=digest(prior['config']) or pred['corpus_id']!=prior['corpus']['corpus_id'] or case['group'] not in pred['groups']:raise ValueError('Matching completed pilots required')
 _,samples=load_corpus(prior['corpus'],a.corpus_root,'profile',mode)
 return s,case,p,samples,pred


async def trial(a):
 s,case,p,samples,pred=inputs(a);a.output.mkdir(parents=True,exist_ok=False)
 os.environ['RAY_USAGE_STATS_ENABLED']='0'
 if p['mode']=='cpu_dummy':
  os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
 import ray
 from ray import serve
 from rosa_repro.fleet_service import deploy_r05_curve
 t=None
 try:
  if p['mode']=='gpu':ray.init(address=a.address,log_to_driver=False)
  else:ray.init(address='local',num_cpus=3,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-curve-',dir='/tmp'),log_to_driver=False)
  serve.start(proxy_location='Disabled');t=deploy_r05_curve(p)
  inventory=await t.inventory();env=await t.environment(p['workers'][0]['worker_id']);backend=inventory[p['workers'][0]['worker_id']]['backend_attestation']
  from rosa_repro.profile_adoption import runtime_signature
  sig=runtime_signature(env,backend) if p['mode']=='gpu' else dict(backend=backend,environment=env)
  if sig!=pred['runtime_signature']:raise ValueError('Curve worker differs from prerequisite pilots')
  r=await run_probe(p,t,samples,case,a.output/'raw',corpus_id=s['pilot_suite']['corpus']['corpus_id'],split='profile');write_json(a.output/'result.json',r)
  return 0 if r['status']=='complete' and r['metrics']['arrivals']==r['metrics']['successful'] else 2
 finally:
  if t:t.close()
  if ray.is_initialized():
   if p['mode']=='cpu_dummy':serve.shutdown()
   ray.shutdown()
  write_json(a.output/'cleanup.json',dict(own_applications_deleted=t is not None,driver_disconnected=True))


def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('prepare');q.add_argument('--pilots-suite',type=Path,required=True);q.add_argument('--rates',type=float,nargs='+',required=True);q.add_argument('--repetitions',type=int,default=3);q.add_argument('--output',type=Path,required=True)
 for name in ('run','_trial'):
  q=sub.add_parser(name)
  for flag in ('suite','pilots','evidence-root','corpus-root','output'):q.add_argument('--'+flag,type=Path,required=True)
  q.add_argument('--case',required=True);q.add_argument('--address')
  if name=='run':q.add_argument('--execute',action='store_true');q.add_argument('--budget-seconds',type=float);q.add_argument('--report',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='_trial':return asyncio.run(trial(a))
 target=a.report if a.cmd=='run' else a.output
 if target.exists():raise FileExistsError(target)
 if a.cmd=='prepare':value=prepare(read(a.pilots_suite),a.rates,a.repetitions)
 else:
  s,case,plan,samples,pred=inputs(a);c=plan['config'];pro=case['protocol']
  budget=pro['warmup_max_s']+pro['measure_max_s']+2*(c['drain_timeout_s']+2*c['request_timeout_s'])+s['pilot_suite']['config']['setup_budget_s']+30
  value=dict(status='planned',minimum_wall_budget_s=budget,models_started=False,case=case)
  if a.execute:
   if a.budget_seconds is None or not math.isfinite(a.budget_seconds) or a.budget_seconds<budget:raise ValueError('Explicit budget must cover declared trial')
   a.output.mkdir(parents=True,exist_ok=False);cmd=[sys.executable,str(Path(__file__).resolve()),'_trial']
   for key in ('suite','pilots','evidence_root','corpus_root'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key).resolve())])
   cmd.extend(['--case',a.case,'--output',str(a.output.resolve()/'trial')])
   if a.address:cmd.extend(['--address',a.address])
   write_json(a.output/'command.json',dict(argv=cmd,budget_seconds=a.budget_seconds));r=supervise(cmd,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),a.output,a.budget_seconds)
   value=dict(status='complete' if r['exit_code']==0 and not r['timed_out'] and not r['interrupted'] else 'failed',process=r,
       needs_cluster_cleanup=plan['mode']=='gpu' and not (a.output/'trial/cleanup.json').exists())
   if value['status']=='complete':value['metrics']=reaudit(a.output/'trial/raw')
 write_json(target,value);return 0 if value.get('status') in (None,'planned','complete') else 2
if __name__=='__main__':raise SystemExit(main())
