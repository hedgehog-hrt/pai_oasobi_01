"""Explicit single-GPU functional trials; separate from reviewed R07 deployment."""
import argparse,asyncio,json,os,sys,tempfile,secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.prototype_plan import prepare,validate
from rosa_repro.closed_loop_protocol import protocol,prepare_suite,validate_suite
from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
from rosa_repro.regression import write_json
from rosa_repro.resume_workflow import supervise
read=lambda p:json.loads(Path(p).read_text())


async def trial(a):
    s=validate_suite(read(a.suite),a.root);plan=validate(s['plan']);p=s['protocol']
    if plan['mode']=='cpu_dummy':
        # Ray snapshots auth configuration on import, before ray.init().
        os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32));os.environ['RAY_AUTH_MODE']='token'
    import ray
    from ray import serve
    from rosa_repro.prototype_service import deploy
    from rosa_repro.simulator_process import ProcessSimulator,PrefetchedSimulator
    if plan['mode']=='gpu' and (not a.address or a.address=='local' or not a.sim_repo):raise ValueError('Explicit GPU cluster/repo required')
    if plan['mode']=='cpu_dummy' and (a.address or a.sim_repo):raise ValueError('CPU fixture refuses external cluster/repo')
    out=a.output;out.mkdir(parents=True,exist_ok=False);transport=None;sims={};clean=False
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    try:
        if plan['mode']=='gpu':ray.init(address=a.address,log_to_driver=False)
        else:
            os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
            ray.init(address='local',num_cpus=3,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',
                _temp_dir=tempfile.mkdtemp(prefix='rosa-prototype-',dir='/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy(plan)
        inventory=await transport.inventory();interfaces={}
        for c in plan['classes']:
            for robot in c['robots']:
                spec=dict(kind='cpu_dummy');wid=robot['routes']['s1']
                if plan['mode']=='gpu':
                    if wid not in interfaces:interfaces[wid]=await transport.simulation_interface(wid)
                    interface=interfaces[wid]
                    if interface['plan_id']!=plan['plan_id'] or interface['instance_id']!=inventory[wid]['instance_id']:
                        raise ValueError('Simulator/model interface identity mismatch')
                    spec=dict(kind='libero',repo=str(Path(a.sim_repo).resolve()),modality_config=interface['modality_config'],
                        max_steps=p['max_steps'],execute_steps=p['execute_steps'],task=s['banks'][c['audit']['id']]['manifest']['task'])
                    if plan.get('collision_guard') and 'safety' in robot['routes']:
                        spec['collision_guard']=plan['collision_guard']
                        if plan.get('recovery_policy'):spec['recovery_policy']=plan['recovery_policy']
                factory=PrefetchedSimulator if plan.get('prefetch_resets') else ProcessSimulator
                sims[robot['robot_id']]=factory(spec,python=a.sim_python,stderr_path=out/f'simulator-{len(sims)}.log',timeout_s=p['sim_timeout_s'])
        write_json(out/'simulation_interfaces.json',interfaces)
        result=await run_closed_loop(s,a.root,transport,sims,out/'raw',rep=a.rep)
        write_json(out/'result.json',result)
        if result['status']=='complete':write_json(out/'reaudit.json',reaudit_closed(out/'raw'))
        return 0 if result['status']=='complete' else 2
    finally:
        for sim in sims.values():
            try:await sim.close()
            except Exception:sim.abort()
        try:
            if transport:
                if hasattr(transport,'aclose'):await transport.aclose()
                else:transport.close()
                clean=True
        finally:
            if ray.is_initialized():
                if plan['mode']=='cpu_dummy':serve.shutdown()
                ray.shutdown()
            write_json(out/'cleanup.json',dict(own_applications_deleted=clean,driver_disconnected=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--configs',type=Path,required=True);q.add_argument('--population',type=Path,required=True)
    q.add_argument('--banks',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--frequency',type=float,default=2.);q.add_argument('--action-time',type=float,default=.4)
    q.add_argument('--warmup',type=float,default=30.);q.add_argument('--duration',type=float,default=30.)
    q.add_argument('--repetitions',type=int,default=1);q.add_argument('--setup-budget',type=float,default=600.)
    q.add_argument('--queue-policy',choices=['fifo','edf_queue_expiry_v1','parallel_roles_v1','parallel_roles_batch_s1_v1'],default='fifo')
    q.add_argument('--collision-guard',type=Path);q.add_argument('--recovery-policy',type=Path)
    q.add_argument('--role-concurrency',type=Path);q.add_argument('--prefetch-resets',action='store_true')
    q.add_argument('--periodic-phases',choices=['synchronized','spread_classes_v1'],default='synchronized')
    for name in ['run','_trial']:
        q=sub.add_parser(name);q.add_argument('--suite',type=Path,required=True);q.add_argument('--root',type=Path,required=True)
        q.add_argument('--output',type=Path,required=True);q.add_argument('--sim-python',required=True);q.add_argument('--sim-repo')
        q.add_argument('--address');q.add_argument('--rep',type=int,default=0)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.command=='prepare':
        plan=prepare({role:read(path) for role,path in read(a.configs).items()},read(a.population),frequency_hz=a.frequency,action_time_s=a.action_time,queue_policy=a.queue_policy,periodic_phases=a.periodic_phases,collision_guard=read(a.collision_guard) if a.collision_guard else None,recovery_policy=read(a.recovery_policy) if a.recovery_policy else None,role_concurrency=read(a.role_concurrency) if a.role_concurrency else None,prefetch_resets=a.prefetch_resets)
        pro=protocol(plan,'gpu_connection' if plan['mode']=='gpu' else 'cpu_smoke',warmup_s=a.warmup,duration_s=a.duration)
        write_json(a.output,prepare_suite(plan,read(a.banks),a.root,pro,repetitions=a.repetitions,setup_budget_s=a.setup_budget));return 0
    if not Path(a.sim_python).is_absolute() or not Path(a.sim_python).is_file():raise ValueError('Explicit simulator Python executable required')
    if a.command=='_trial':return asyncio.run(trial(a))
    s=validate_suite(read(a.suite),a.root);validate(s['plan'])
    if not 0<=a.rep<s['repetitions']:raise ValueError('Unknown repetition')
    a.output.mkdir(parents=True)
    cmd=[sys.executable,str(Path(__file__).resolve()),'_trial','--suite',str(a.suite.resolve()),'--root',str(a.root.resolve()),
        '--output',str(a.output.resolve()/'trial'),'--sim-python',a.sim_python,'--rep',str(a.rep)]
    for flag,value in [('--address',a.address),('--sim-repo',a.sim_repo)]:
        if value:cmd.extend([flag,value])
    write_json(a.output/'command.json',dict(argv=cmd,wall_budget_s=s['trial_wall_budget_s']))
    process=supervise(cmd,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),a.output,s['trial_wall_budget_s'])
    result=read(a.output/'trial/result.json') if (a.output/'trial/result.json').is_file() else {}
    cleanup=read(a.output/'trial/cleanup.json') if (a.output/'trial/cleanup.json').is_file() else {}
    passed=process['exit_code']==0 and result.get('status')=='complete' and cleanup.get('own_applications_deleted')
    write_json(a.output/'result.json',dict(status='complete' if passed else 'failed',process=process,trial=result,cleanup=cleanup,
        formal_performance_claim=False,semantic_quality_verified=False))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
