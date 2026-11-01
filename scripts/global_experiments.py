"""Bounded dynamic R07 trials. GPU mode connects only to an explicitly named existing cluster."""
import argparse,asyncio,os,secrets,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.profile_adoption import read,inside
from rosa_repro.regression import write_json,sha256
from rosa_repro.global_scenario import prepare,validate
from rosa_repro.global_control import Supervisor
from rosa_repro.global_planner import ProcessPlanner
from rosa_repro.global_managers import RayManager
from rosa_repro.global_health import watch
from rosa_repro.global_audit import evaluate,seal_tree,verify_tree
from rosa_repro.dynamic_runner import run
from rosa_repro.paired import load_state_bank
from rosa_repro.resume_workflow import supervise


async def trial(a):
    from rosa_repro.simulator_process import ProcessSimulator
    s=validate(read(a.scenario),a.root);gpu=s['spec']['catalog']['mode']=='gpu';out=a.output;out.mkdir(parents=True,exist_ok=False)
    if gpu and (not a.address or a.address=='local' or not a.sim_repo):raise ValueError('GPU requires an existing cluster and pinned LIBERO repo')
    if not gpu and (a.address or a.sim_repo):raise ValueError('CPU trial refuses external cluster/repo')
    planner=ProcessPlanner(a.solver_python);manager=RayManager(read(a.adoptions) if a.adoptions else None)
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    if not gpu:os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    import ray
    from ray import serve
    controller=None;sims={};stop=asyncio.Event();health=None;cleanup=False
    rep=getattr(a,'rep',0)
    if type(rep) is not int or not 0<=rep<s['repetitions']:raise ValueError('Invalid dynamic repetition')
    write_json(out/'scenario.json',s);write_json(out/'trial_condition.json',dict(scenario_id=s['id'],rep=rep))
    try:
        if gpu:ray.init(address=a.address,log_to_driver=False)
        else:ray.init(address='local',num_cpus=max([s['spec']['gpu_budget']]+[e['gpu_budget'] for e in s['changes'] if e['op']=='budget'])+2,num_gpus=0,
            include_dashboard=False,_node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-global-',dir='/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');controller=Supervisor(s['spec'],manager,planner,out/'control')
        if (await controller.start())['status']!='committed':raise ValueError('Initial placement unresolved')
        banks={c:load_state_bank(inside(a.root,b['directory'])) for c,b in s['banks'].items()};initial={};interfaces={};classes={c['id']:c for c in s['spec']['catalog']['fleet']['classes']}
        for i,(rid,cid) in enumerate(s['robots'].items()):
            m,states=banks[cid];case=m['cases'][(i+rep)%len(m['cases'])];initial[rid]=dict(class_id=cid,state=states[case['case_id']],seed=case['seed'])
            spec=dict(kind='cpu_dummy',max_steps=s['max_steps'])
            if gpu:
                mid=next(m['id'] for m in classes[cid]['components'] if m['role']=='s1')
                candidates=[w for w in controller.current.plan['workers'] if w['component_id']==mid]
                if not candidates:raise ValueError('GPU simulator requires this class S1 to be deployed initially')
                w=candidates[0];interface=await asyncio.wait_for(manager.live.simulation_interface(w['worker_id']),s['spec']['call_timeout_s'])
                if interface['plan_id']!=controller.current.plan['plan_id'] or interface['instance_id']!=controller.current.inventory[w['worker_id']]['instance_id']:raise ValueError('Simulator/model identity differs')
                interfaces[rid]=interface;spec=dict(kind='libero',repo=str(Path(a.sim_repo).resolve()),modality_config=interface['modality_config'],max_steps=s['max_steps'],execute_steps=8,task=m['task'])
            sims[rid]=ProcessSimulator(spec,python=a.sim_python,stderr_path=out/f'simulator-{i}.log',timeout_s=s['spec']['call_timeout_s'])
        write_json(out/'simulation_interfaces.json',interfaces)
        async def change(e):
            op=e['op']
            if op=='join':return await controller.join(e['class_id'],e['robot_id'])
            if op=='leave':return await controller.leave(e['robot_id'])
            if op=='budget':return await controller.reconcile(None,gpu_budget=e['gpu_budget'],reason='budget_change')
            if op=='replan':return await controller.reconcile(None,reason='requested_replan')
            if op=='fault':
                wid=controller.current.routes[e['robot_id']][e['component_id']];failed_epoch=controller.epoch
                await asyncio.to_thread(manager.live.fail_worker,wid);controller.emit('owned_actor_fault_injected',worker_id=wid,epoch=failed_epoch)
                return dict(status='injected',worker_id=wid,failed_epoch=failed_epoch)
        health=asyncio.create_task(watch(controller,stop,**s['health']))
        changes=[(e['offset_s'],lambda e=e:change(e)) for e in s['changes']]
        await run(controller,sims,initial,changes,warmup_s=s['warmup_s'],duration_s=s['duration_s'],output=out/'simulation',
            sim_timeout_s=s['spec']['call_timeout_s'],max_requests=s['max_requests'],max_capture_bytes=s['max_capture_bytes'],evaluation=s['evaluation'])
        stop.set();await health;await controller.close();result=evaluate(out/'control');write_json(out/'evaluation.json',result)
        return 0 if result['status']=='complete' else 2
    finally:
        stop.set()
        if health and not health.done():
            try:await asyncio.wait_for(health,s['spec']['deployment_timeout_s']+s['spec']['call_timeout_s'])
            except Exception:pass
        for sim in sims.values():
            try:await sim.close()
            except Exception:sim.abort()
        try:
            if controller:await controller.close()
            cleanup=manager.live is None and manager.pending is None
        finally:
            if ray.is_initialized():
                if not gpu:serve.shutdown()
                ray.shutdown()
            write_json(out/'cleanup.json',dict(own_fleet_cleanup_confirmed=cleanup,driver_disconnected=True));seal_tree(out)


def execute(a):
    s=validate(read(a.scenario),a.root)
    if not 0<=a.rep<s['repetitions']:raise ValueError('Invalid dynamic repetition')
    if not a.execute:return dict(status='planned',scenario_id=s['id'],rep=a.rep,wall_budget_s=s['wall_budget_s'],models_started=False)
    if a.budget_seconds is None or not s['wall_budget_s']<=a.budget_seconds<=86400:raise ValueError('Explicit wall budget must cover the whole trial')
    # Fresh output only: uncertain GPU cleanup never triggers an automatic retry.
    a.output.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(Path(__file__).resolve()),'_trial','--scenario',str(a.scenario.resolve()),'--root',str(a.root.resolve()),'--output',str(a.output/'trial'),
        '--solver-python',a.solver_python,'--sim-python',a.sim_python,'--rep',str(a.rep)]
    for flag,value in (('--address',a.address),('--sim-repo',a.sim_repo),('--adoptions',str(a.adoptions.resolve()) if a.adoptions else None)):
        if value:cmd.extend([flag,value])
    for path in (a.solver_python,a.sim_python):
        if not Path(path).is_absolute() or not Path(path).is_file():raise ValueError('Pinned interpreter path required')
    write_json(a.output/'command.json',dict(argv=cmd,source_manifest=s['source_manifest'],solver_sha256=sha256(Path(a.solver_python).resolve()),simulator_sha256=sha256(Path(a.sim_python).resolve())))
    process=supervise(cmd,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),a.output,s['wall_budget_s'])
    cleanup=read(a.output/'trial/cleanup.json') if (a.output/'trial/cleanup.json').exists() else {}
    complete=process['exit_code']==0 and not process['timed_out'] and cleanup.get('own_fleet_cleanup_confirmed') is True
    if complete:
        from rosa_repro.dynamic_evidence import reaudit
        checked=reaudit(a.output/'trial')
        if checked['status']!='complete':raise ValueError('Dynamic evidence incomplete')
        if s['evaluation'] is not None:
            from rosa_repro.dynamic_protocol import inspect_trial
            acceptance=inspect_trial(a.output/'trial');write_json(a.output/'acceptance.json',acceptance)
            complete=acceptance['status']=='accepted'
    result=dict(status='complete' if complete else 'failed',process=process,needs_cluster_cleanup=s['spec']['catalog']['mode']=='gpu' and cleanup.get('own_fleet_cleanup_confirmed') is not True)
    write_json(a.output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--spec',type=Path,required=True);q.add_argument('--banks',type=Path,required=True);q.add_argument('--changes',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--warmup',type=float,default=1.);q.add_argument('--duration',type=float,default=120.);q.add_argument('--setup-budget',type=float,default=600.)
    q.add_argument('--phase',choices=['diagnostic','cpu_smoke','formal'],default='diagnostic');q.add_argument('--repetitions',type=int,default=3)
    q.add_argument('--warmup-min-success',type=int);q.add_argument('--measure-min-success',type=int,default=1);q.add_argument('--change-tolerance',type=float,default=1.)
    for cmd in ('run','_trial'):
        q=sub.add_parser(cmd)
        for key in ('scenario','root','output'):q.add_argument('--'+key,type=Path,required=True)
        q.add_argument('--solver-python',required=True);q.add_argument('--sim-python',required=True);q.add_argument('--address');q.add_argument('--sim-repo');q.add_argument('--adoptions',type=Path)
        q.add_argument('--rep',type=int,default=0)
        if cmd=='run':q.add_argument('--execute',action='store_true');q.add_argument('--budget-seconds',type=float);q.add_argument('--report',type=Path,required=True)
    q=sub.add_parser('reaudit');q.add_argument('directory',type=Path);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('report');q.add_argument('--scenario',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--directories',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='_trial':return asyncio.run(trial(a))
    target=a.report if a.command=='run' else a.output
    if target.exists():raise FileExistsError(target)
    if a.command=='prepare':
        from rosa_repro.dynamic_protocol import protocol
        spec=read(a.spec);policy=None if a.phase=='diagnostic' else protocol(spec,phase=a.phase,warmup_s=a.warmup,duration_s=a.duration,warmup_min_success=a.warmup_min_success,measure_min_success=a.measure_min_success,change_tolerance_s=a.change_tolerance)
        value=prepare(spec,read(a.banks),a.root,read(a.changes),warmup_s=a.warmup,duration_s=a.duration,setup_budget_s=a.setup_budget,evaluation=policy,repetitions=a.repetitions)
    elif a.command=='report':
        from rosa_repro.dynamic_protocol import report
        value=report(read(a.scenario),a.root,read(a.directories))
    elif a.command=='reaudit':
        from rosa_repro.dynamic_evidence import reaudit
        value=reaudit(a.directory)
    else:value=execute(a)
    write_json(target,value);return 0 if value.get('status') in (None,'planned','complete') else 2
if __name__=='__main__':raise SystemExit(main())
