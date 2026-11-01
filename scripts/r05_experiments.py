"""SLO-free R05 generation, bounded local/Pod execution, resume and raw comparison."""
import argparse,asyncio,json,os,secrets,sys,tempfile,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.r05_protocol import template,prepare,validate_suite,plan_for,timing
from rosa_repro.r05_runtime import run_trial,reaudit
from rosa_repro.r05_workflow import check_pilots,report
from rosa_repro.regression import write_json,sha256
from rosa_repro.execution_plan import digest
from rosa_repro.experiments import ExperimentStore
from rosa_repro.resume_workflow import supervise
read=lambda p:json.loads(Path(p).read_text())


async def trial(a):
    s=validate_suite(read(a.suite),a.corpus_root);c=s['config']
    if not s['executable']:raise ValueError('Unresolved timing/sample budget')
    deps=check_pilots(s,a.evidence_root,read(a.pilots) if a.pilots else None)
    cell=next(x for x in s['cells'] if x['id']==a.cell);plan=plan_for(c,cell)
    if (c['mode']=='gpu' and (not a.address or a.address=='local')) or (c['mode']=='cpu_dummy' and a.address):raise ValueError('Explicit existing GPU cluster only; CPU uses local cluster')
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_r05
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    if c['mode']=='cpu_dummy':
        os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32))
        os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    out=a.output;out.mkdir(parents=True,exist_ok=False);transport=None;clean=False
    write_json(out/'pilot_dependencies.json',deps)
    try:
        if c['mode']=='gpu':ray.init(address=a.address,log_to_driver=False)
        else:ray.init(address='local',num_cpus=3,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',
                      _temp_dir=tempfile.mkdtemp(prefix='rosa-r05-',dir='/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy_r05(plan)
        result=await run_trial(s,a.corpus_root,a.cell,transport,out/'raw',pilot_runtime=deps.get(cell['group'],{}).get('backend_attestation'))
        return 0 if result['status']=='complete' else 2
    finally:
        try:
            if transport:transport.close();clean=True
        finally:
            if ray.is_initialized():
                if c['mode']=='cpu_dummy':serve.shutdown()
                ray.shutdown()
            write_json(out/'cleanup.json',dict(own_deployments_removed=clean,driver_disconnected=True))


def execute(a):
    s=validate_suite(read(a.suite),a.corpus_root);c=s['config'];budget=s['phase_budget_s']+c['setup_budget_s']+4*max(5.,c['watchdog_s'])+30
    cases=[x for x in s['cells'] if a.cell is None or x['id']==a.cell]
    if not cases:raise ValueError('Unknown R05 cell')
    if not a.execute:return dict(status='planned',suite_id=s['suite_id'],cases=cases,executable=s['executable'],blocking_reasons=s['blocking_reasons'],
        maximum_one_trial_wall_s=budget,models_started=False,latency_slo_ms=None)
    if not s['executable']:raise ValueError('Extend fixed timing/sample budget before execution')
    if type(a.budget_seconds) not in (int,float) or not math.isfinite(a.budget_seconds) or a.budget_seconds<=0 or a.max_new<1:raise ValueError('Positive explicit wall budget/max-new required')
    if c['mode']=='gpu' and (not a.address or a.address=='local'):raise ValueError('Existing GPU Ray address required')
    if c['mode']=='cpu_dummy' and a.address:raise ValueError('CPU cannot use external cluster')
    deps=check_pilots(s,a.evidence_root,read(a.pilots) if a.pilots else None)
    matrix=dict(suite=s,driver_python=sys.executable,python_binary_sha256=sha256(Path(sys.executable).resolve()),address=a.address,dependencies=deps)
    start=time.monotonic();count=0;rows=[]
    with ExperimentStore(a.output,matrix) as store:
        for case in cases:
            cid=case['id'];old=store.db.execute('SELECT number,status FROM attempts WHERE case_id=? ORDER BY number DESC LIMIT 1',(cid,)).fetchone()
            if old:
                prev=a.output/cid/f'attempt-{old[0]:03d}';r=read(prev/'result.json') if (prev/'result.json').exists() else {}
                if old[1]=='complete' and store.completed(cid) and r.get('status')=='complete':
                    reaudit(prev/'trial/raw');rows.append(dict(cell_id=cid,status='skipped'));continue
                if c['mode']=='gpu' and (old[1]!='complete' or r.get('needs_cluster_cleanup')):
                    token=digest(dict(cell_id=cid,attempt=old[0],files={str(p.relative_to(prev)):sha256(p) for p in prev.rglob('*') if p.is_file()}))
                    review=read(a.cleanup_review) if a.cleanup_review else {}
                    if review.get('attempt_sha256')!=token or review.get('cleanup_confirmed') is not True or not all(isinstance(review.get(k),str) and review[k].strip() for k in ('reviewer','reason')):
                        return dict(status='blocked_cluster_cleanup',attempt_sha256=token,rows=rows,executed_trials=count)
            if count>=a.max_new:return dict(status='trial_limit',rows=rows,executed_trials=count)
            if time.monotonic()-start+budget>a.budget_seconds:return dict(status='budget_stop',rows=rows,executed_trials=count,next_cell=cid)
            with store.attempt(cid) as out:
                cmd=[sys.executable,str(Path(__file__).resolve()),'_trial','--suite',str(a.suite.resolve()),'--corpus-root',str(a.corpus_root.resolve()),
                    '--cell',cid,'--output',str(out.resolve()/'trial'),'--evidence-root',str(a.evidence_root.resolve())]
                for flag,value in (('--address',a.address),('--pilots',str(a.pilots.resolve()) if a.pilots else None)):
                    if value:cmd+=[flag,value]
                write_json(out/'command.json',dict(argv=cmd,wall_budget_s=budget))
                if a.cleanup_review:write_json(out/'cleanup_review.json',read(a.cleanup_review))
                process=supervise(cmd,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),out,budget)
                cleanup=read(out/'trial/cleanup.json') if (out/'trial/cleanup.json').exists() else {}
                audit=reaudit(out/'trial/raw') if (out/'trial/raw/evidence.json').exists() else None
                r=dict(status='complete' if process['exit_code']==0 and audit and audit['status']=='complete' and cleanup.get('own_deployments_removed') else 'failed',
                    process=process,audit=audit,needs_cluster_cleanup=c['mode']=='gpu' and (process['timed_out'] or process['interrupted'] or process['exit_code']<0 or not cleanup.get('own_deployments_removed') or not audit or not audit['drained']))
                write_json(out/'result.json',r)
            count+=1;rows.append(dict(cell_id=cid,status=r['status'],directory=str(out)))
            if r['status']!='complete':return dict(status='failed',rows=rows,executed_trials=count)
    return dict(status='complete',rows=rows,executed_trials=count)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('predict');q.add_argument('--profiles',type=Path);q.add_argument('--pilots',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--tie-tolerance',type=float,default=.01);q.add_argument('--selection-output',type=Path);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('template');q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('--paper',type=Path,required=True);q.add_argument('--config',type=Path,required=True);q.add_argument('--corpus',type=Path,required=True)
    q.add_argument('--stage',choices=['pilot','confirmation'],required=True);q.add_argument('--group',action='append');q.add_argument('--protocol',type=Path);q.add_argument('--prediction-evidence',type=Path,action='append',default=[]);q.add_argument('--output',type=Path,required=True)
    for name in ('run','_trial'):
        q=sub.add_parser(name);q.add_argument('--suite',type=Path,required=True);q.add_argument('--corpus-root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        q.add_argument('--cell');q.add_argument('--address');q.add_argument('--pilots',type=Path);q.add_argument('--evidence-root',type=Path,default=Path('.'))
        if name=='run':
            q.add_argument('--execute',action='store_true');q.add_argument('--max-new',type=int,default=1);q.add_argument('--budget-seconds',type=float);q.add_argument('--report',type=Path,required=True);q.add_argument('--cleanup-review',type=Path)
    q=sub.add_parser('reaudit');q.add_argument('--raw',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('report');q.add_argument('--suite',type=Path,required=True);q.add_argument('--directories',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='_trial':return asyncio.run(trial(a))
    target=a.report if a.command=='run' else a.output
    if target.exists():raise FileExistsError(target)
    if a.command=='predict' and a.selection_output and (a.selection_output.exists() or a.selection_output.resolve()==target.resolve()):raise ValueError('Fresh distinct selection output required')
    if a.command=='predict':
        from rosa_repro.r05_prediction import generate
        if a.profiles:
            from rosa_repro.r05_load_curves import generate as from_curves
            value=from_curves(a.root,read(a.pilots),read(a.profiles),relative_tie_tolerance=a.tie_tolerance)
        else:value=generate(a.root,read(a.pilots),relative_tie_tolerance=a.tie_tolerance)
    elif a.command=='template':value=template()
    elif a.command=='prepare':value=prepare(read(a.paper),read(a.config),read(a.corpus),stage=a.stage,groups=a.group,protocol=read(a.protocol) if a.protocol else None,
        prediction_artifacts={sha256(p):p.read_text() for p in a.prediction_evidence})
    elif a.command=='run':value=execute(a)
    elif a.command=='reaudit':value=reaudit(a.raw)
    else:value=report(validate_suite(read(a.suite)),a.root,read(a.directories))
    write_json(target,value)
    if a.command=='predict' and a.selection_output:
        write_json(a.selection_output,{g:dict(batch_size=b,prediction_evidence_sha256=sha256(target)) for g,b in value['groups'].items()})
    return 0 if value.get('status') in (None,'complete','planned','trial_limit') else 2
if __name__=='__main__':raise SystemExit(main())
