"""Prepare R10 specs and supervise CPU Ray process-fault proofs, with explicit budget."""
import argparse,asyncio,json,os,sys,secrets,tempfile,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.dynamic_control import prepare,validate,Controller,reaudit
from rosa_repro.control_scenario import validate_scenario,run
from rosa_repro.regression import write_json
from rosa_repro.resume_workflow import supervise

def read(p):return json.loads(Path(p).read_text())

async def trial(a):
    spec=validate(read(a.spec));steps=validate_scenario(spec,read(a.scenario));out=a.output;out.mkdir(parents=True,exist_ok=False)
    os.environ['RAY_USAGE_STATS_ENABLED']='0';os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32))
    os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_control
    transport=None;controller=None
    try:
        ray.init(address='local',num_cpus=len(spec['workers'])+2,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-r10-',dir='/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy_control(spec);controller=Controller(spec,transport,out/'raw')
        await controller.start();write_json(out/'scenario.json',steps)
        rows=await run(controller,transport,steps);result=await controller.close();write_json(out/'scenario_result.json',dict(steps=rows,control=result))
        return 0 if result['status']=='complete' else 2
    finally:
        if controller and not controller.closed:await controller.close()
        try:
            if transport:transport.close()
        finally:
            if ray.is_initialized():serve.shutdown();ray.shutdown()
        write_json(out/'cleanup.json',dict(own_applications_deleted=transport is not None,driver_disconnected=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--plan',type=Path,required=True);q.add_argument('--initial',type=Path,required=True);q.add_argument('--standbys',type=Path,required=True)
    q.add_argument('--gpu-budget',type=int,required=True);q.add_argument('--timeout',type=float,default=5.);q.add_argument('--output',type=Path,required=True)
    for cmd in ('run','_trial'):
        q=sub.add_parser(cmd);q.add_argument('--spec',type=Path,required=True);q.add_argument('--scenario',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if cmd=='run':q.add_argument('--execute',action='store_true');q.add_argument('--budget-seconds',type=float);q.add_argument('--report',type=Path,required=True)
    q=sub.add_parser('reaudit');q.add_argument('directory',type=Path);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='_trial':return asyncio.run(trial(a))
    target=a.report if a.cmd=='run' else a.output
    if target.exists():raise FileExistsError(target)
    if a.cmd=='prepare':value=prepare(read(a.plan),read(a.initial),read(a.standbys),physical_gpu_budget=a.gpu_budget,timeout_s=a.timeout)
    elif a.cmd=='reaudit':value=reaudit(a.directory)
    else:
        spec=validate(read(a.spec));steps=validate_scenario(spec,read(a.scenario))
        worst=120+sum(s.get('seconds',0)+spec['timeout_s']*(len(spec['workers'])+3)+max(c['action_time_s'] for c in spec['blueprint']['classes']) for s in steps)
        value=dict(status='planned',steps=len(steps),minimum_wall_budget_s=worst,models_started=False,mode='cpu_dummy')
        if a.execute:
            if a.budget_seconds is None or not math.isfinite(a.budget_seconds) or a.budget_seconds<worst:raise ValueError('Budget must cover setup and all bounded scenario calls')
            a.output.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,str(Path(__file__).resolve()),'_trial','--spec',str(a.spec.resolve()),'--scenario',str(a.scenario.resolve()),'--output',str(a.output.resolve()/'trial')]
            write_json(a.output/'command.json',dict(argv=command,budget_seconds=a.budget_seconds))
            proc=supervise(command,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),a.output,a.budget_seconds)
            value=dict(status='complete' if proc['exit_code']==0 and not proc['timed_out'] and not proc['interrupted'] else 'failed',process=proc,mode='cpu_dummy')
            if value['status']=='complete':value['control']=reaudit(a.output/'trial/raw')
    write_json(target,value);return 0 if value.get('status') in (None,'planned','complete') else 2
if __name__=='__main__':raise SystemExit(main())
