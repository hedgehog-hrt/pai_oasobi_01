"""Prepare and supervise fixed-window live P1-P4/R07 trials; no Pod lifecycle calls."""
import argparse,asyncio,json,os,sys,tempfile,secrets,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.closed_loop_protocol import protocol,prepare_suite,validate_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.profile_adoption import read,validate_fleet_adoptions
from rosa_repro.regression import write_json,sha256
from rosa_repro.resume_workflow import supervise
from rosa_repro.experiments import ExperimentStore
from rosa_repro.execution_plan import runtime_source_manifest,digest


async def trial(a):
    from rosa_repro.simulator_process import ProcessSimulator
    s=validate_suite(read(a.suite),a.root);plan=s['plan'];p=s['protocol']
    adoptions=read(a.adoptions) if a.adoptions else None;validate_fleet_adoptions(plan,adoptions)
    if plan['mode']=='gpu' and (not a.address or a.address=='local' or not a.sim_repo):raise ValueError('GPU requires existing Ray address and pinned simulator repo')
    if plan['mode']=='cpu_dummy' and (a.address or a.sim_repo):raise ValueError('CPU cannot use external cluster or real simulator')
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    if plan['mode']=='cpu_dummy':
        os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32))
        os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_fleet
    transport=None;sims={};out=a.output;out.mkdir(parents=True,exist_ok=False)
    try:
        if plan['mode']=='gpu':ray.init(address=a.address,log_to_driver=False)
        else:ray.init(address='local',num_cpus=len(plan['workers'])+2,num_gpus=0,include_dashboard=False,
            _node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-live-',dir='/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy_fleet(plan,adoptions=adoptions,residency_calibration=a.residency_calibration,joint_review=read(a.joint_review) if a.joint_review else None)
        inventory=await transport.inventory();interfaces={}
        for c in plan['classes']:
            for robot in c['robots']:
                rid=robot['robot_id'];spec=dict(kind='cpu_dummy')
                if plan['mode']=='gpu':
                    mid=next(x['id'] for x in c['audit']['components'] if x['role']=='s1');wid=robot['routes'][mid]
                    if wid not in interfaces:interfaces[wid]=await asyncio.wait_for(transport.simulation_interface(wid),p['request_timeout_s'])
                    interface=interfaces[wid]
                    if interface['plan_id']!=plan['plan_id'] or interface['worker_id']!=wid or interface['instance_id']!=inventory[wid]['instance_id']:
                        raise ValueError('Simulator interface/model identity mismatch')
                    spec=dict(kind='libero',repo=str(Path(a.sim_repo).resolve()),modality_config=interface['modality_config'],
                        max_steps=p['max_steps'],execute_steps=p['execute_steps'],task=s['banks'][c['audit']['id']]['manifest']['task'])
                sims[rid]=ProcessSimulator(spec,python=a.sim_python,stderr_path=out/f'simulator-{len(sims)}.log',timeout_s=p['sim_timeout_s'])
        write_json(out/'simulation_interfaces.json',interfaces)
        result=await run_closed_loop(s,a.root,transport,sims,out/'raw',rep=a.rep)
        write_json(out/'result.json',result);return 0 if result['status']=='complete' else 2
    finally:
        for sim in sims.values():
            try:await sim.close()
            except Exception:sim.abort()
        try:
            if transport:transport.close()
        finally:
            if ray.is_initialized():
                if plan['mode']=='cpu_dummy':serve.shutdown()
                ray.shutdown()
        write_json(out/'cleanup.json',dict(status='complete' if transport is not None else 'deployment_not_confirmed',
            own_serve_applications_deleted=transport is not None,driver_disconnected=True))


def execute(a):
    s=validate_suite(read(a.suite),a.root);plan=s['plan'];budget=s['trial_wall_budget_s']+10
    cases=[c for c in s['cases'] if not a.case_id or c['case_id']==a.case_id]
    if not cases:raise ValueError('Unknown requested case')
    if not a.execute:return dict(status='planned',cases=cases,trial_wall_budget_s=budget,models_started=False)
    if type(a.budget_seconds) not in (float,int) or not math.isfinite(a.budget_seconds) or a.budget_seconds<=0 or a.max_new<1:raise ValueError('Explicit positive wall budget and max-new required')
    validate_fleet_adoptions(plan,read(a.adoptions) if a.adoptions else None)
    if plan['mode']=='gpu' and (not a.address or a.address=='local' or not a.sim_repo):raise ValueError('GPU requires explicit address and simulator repo')
    if plan['mode']=='cpu_dummy' and (a.address or a.sim_repo):raise ValueError('CPU refuses external cluster/repo')
    if not Path(a.sim_python).is_absolute() or not Path(a.sim_python).is_file():raise ValueError('Explicit simulator Python executable required')
    import time
    start=time.monotonic();count=0;rows=[];output=a.output.resolve()
    matrix=dict(suite=s,driver_python=sys.executable,driver_python_sha256=sha256(Path(sys.executable).resolve()),
        sim_python=a.sim_python,sim_python_sha256=sha256(Path(a.sim_python).resolve()),sim_repo=a.sim_repo,address=a.address,
        adoptions_sha256=sha256(a.adoptions) if a.adoptions else None,residency_calibration=a.residency_calibration,joint_review_sha256=sha256(a.joint_review) if a.joint_review else None)
    with ExperimentStore(output,matrix) as store:
        for case in cases:
            cid=case['case_id'];latest=store.db.execute('SELECT number,status FROM attempts WHERE case_id=? ORDER BY number DESC LIMIT 1',(cid,)).fetchone()
            if latest:
                previous=output/cid/f'attempt-{latest[0]:03d}';old=read(previous/'result.json') if (previous/'result.json').exists() else {}
                if latest[1]=='complete' and store.completed(cid) and old.get('status')=='complete':
                    reaudit_closed(previous/'trial/raw');rows.append(dict(case_id=cid,status='skipped'));continue
                if plan['mode']=='gpu' and (latest[1]!='complete' or old.get('needs_cluster_cleanup')):
                    token=digest(dict(case_id=cid,attempt=latest[0],files={str(x.relative_to(previous)):sha256(x) for x in previous.rglob('*') if x.is_file()}))
                    review=read(a.cleanup_review) if a.cleanup_review else {}
                    if review.get('attempt_sha256')!=token or review.get('cleanup_confirmed') is not True or not all(isinstance(review.get(k),str) and review[k].strip() for k in ('reviewer','reason')):
                        return dict(status='blocked_cluster_cleanup',attempt_sha256=token,rows=rows,executed_trials=count)
            if count>=a.max_new:return dict(status='trial_limit',rows=rows,executed_trials=count)
            if time.monotonic()-start+budget>a.budget_seconds:return dict(status='budget_stop',next_case=cid,rows=rows,executed_trials=count)
            with store.attempt(cid) as directory:
                command=[sys.executable,str(Path(__file__).resolve()),'_trial','--suite',str(a.suite.resolve()),'--root',str(a.root.resolve()),
                    '--output',str(directory/'trial'),'--rep',str(case['rep']),'--sim-python',a.sim_python]
                for flag,value in (('--address',a.address),('--adoptions',str(a.adoptions.resolve()) if a.adoptions else None),('--sim-repo',a.sim_repo)):
                    if value:command.extend([flag,value])
                if a.residency_calibration:command.append('--residency-calibration')
                if a.joint_review:command.extend(['--joint-review',str(a.joint_review.resolve())])
                write_json(directory/'command.json',dict(argv=command,wall_budget_s=s['trial_wall_budget_s']))
                if a.cleanup_review:write_json(directory/'cleanup_review.json',read(a.cleanup_review))
                process=supervise(command,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),directory,s['trial_wall_budget_s'])
                cleanup=read(directory/'trial/cleanup.json') if (directory/'trial/cleanup.json').exists() else {}
                r=dict(status='failed',process=process,needs_cluster_cleanup=plan['mode']=='gpu' and
                    (process['timed_out'] or process['interrupted'] or process['exit_code']<0 or cleanup.get('status')!='complete'))
                if (directory/'trial/result.json').exists():r['trial_result']=read(directory/'trial/result.json')
                if process['exit_code']==0 and not process['timed_out'] and not process['interrupted']:
                    reaudit_closed(directory/'trial/raw')
                    if r['trial_result']['status']=='complete':r['status']='complete'
                write_json(directory/'result.json',r)
            count+=1;rows.append(dict(case_id=cid,status=r['status'],directory=str(directory)))
            if r['status']!='complete':return dict(status='failed',rows=rows,executed_trials=count)
    return dict(status='complete',rows=rows,executed_trials=count)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--plan',type=Path,required=True);q.add_argument('--banks',type=Path,required=True);q.add_argument('--root',type=Path,required=True)
    q.add_argument('--phase',choices=['formal','cpu_smoke'],default='formal');q.add_argument('--warmup',type=float);q.add_argument('--duration',type=float)
    q.add_argument('--warmup-min-success',type=int)
    q.add_argument('--protocol',type=Path);q.add_argument('--repetitions',type=int,default=3);q.add_argument('--setup-budget',type=float,default=600);q.add_argument('--output',type=Path,required=True)
    for name in ('run','_trial'):
        q=sub.add_parser(name);q.add_argument('--suite',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        q.add_argument('--address');q.add_argument('--adoptions',type=Path);q.add_argument('--sim-python',default=sys.executable);q.add_argument('--sim-repo')
        q.add_argument('--residency-calibration',action='store_true');q.add_argument('--joint-review',type=Path)
        if name=='run':
            q.add_argument('--execute',action='store_true');q.add_argument('--max-new',type=int,default=1);q.add_argument('--budget-seconds',type=float)
            q.add_argument('--report',type=Path,required=True);q.add_argument('--cleanup-review',type=Path);q.add_argument('--case-id')
        else:q.add_argument('--rep',type=int,required=True)
    q=sub.add_parser('reaudit');q.add_argument('directory',type=Path);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='_trial':return asyncio.run(trial(a))
    target=a.report if a.command=='run' else a.output
    if target.exists():raise FileExistsError(target)
    if a.command=='prepare':
        plan=read(a.plan);pro=read(a.protocol) if a.protocol else protocol(plan,a.phase,warmup_s=a.warmup,duration_s=a.duration,warmup_min_success=a.warmup_min_success)
        result=prepare_suite(plan,read(a.banks),a.root,pro,repetitions=a.repetitions,setup_budget_s=a.setup_budget)
    elif a.command=='reaudit':result=reaudit_closed(a.directory)
    else:result=execute(a)
    write_json(target,result);return 0 if result.get('status') in (None,'planned','complete','trial_limit') else 2
if __name__=='__main__':raise SystemExit(main())
