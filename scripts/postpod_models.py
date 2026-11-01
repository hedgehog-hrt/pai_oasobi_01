"""Prepare and execute saved-observation component probes. Never manages a Pod.
GPU execution requires an explicit existing Ray address and resolved configuration.
"""
import argparse,asyncio,json,os,secrets,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.contracts import load_json
from rosa_repro.execution_plan import digest
from rosa_repro.experiments import ExperimentStore
from rosa_repro.profile_protocol import (template,validate_probe_config,probe_plan,build_corpus,load_corpus,
    make_suite,validate_suite,preflight,environment_evidence)
from rosa_repro.profile_runner import run_probe,reaudit,profile_candidate,verify_evidence,pilot_rate_hints,compare_probes
from rosa_repro.regression import write_json,sha256


def check_smoke(directory,suite):
    p=Path(directory);verify_evidence(p);m=reaudit(p)
    x=load_json(p/'condition.json');r=load_json(p/'result.json')
    if x['config']!=suite['config'] or x['corpus_id']!=suite['corpus']['corpus_id'] or x['split']!='evaluation' or x['case']['stage']!='smoke':
        raise ValueError('Smoke evidence does not match model configuration/corpus')
    if r['status']!='complete' or not r['drained'] or m['arrivals']!=m['successful'] or not m['successful']:
        raise ValueError('Smoke probe contains errors or shortfall')
    if load_json(p/'source_manifest.json')!=suite['source_manifest']:raise ValueError('Smoke source version differs')
    return dict(path=str(p.resolve()),evidence_sha256=sha256(p/'evidence.json'),semantic_accuracy_verified=False)


async def execute(suite,root,corpus_root,*,address=None,max_new=1,smoke_evidence=None,case_id=None):
    validate_suite(suite);c=suite['config'];stage=suite['cases'][0]['stage']
    if not suite['executable']:raise ValueError('Suite has unresolved timing/budget conditions; no model will be started')
    if max_new<0:raise ValueError('max-new must be nonnegative')
    if case_id is not None and case_id not in {x['case_id'] for x in suite['cases']}:
        raise ValueError('Unknown requested case')
    if c['mode']=='gpu' and (not address or address=='local'):raise ValueError('GPU run requires an explicit existing Ray cluster address')
    if c['mode']=='cpu_dummy' and address:raise ValueError('CPU harness cannot connect to an external cluster')
    corpus,samples=load_corpus(suite['corpus'],corpus_root,suite['split'],c['mode'])
    smoke=check_smoke(smoke_evidence,suite) if smoke_evidence else None
    if c['mode']=='gpu' and stage!='smoke' and not smoke:raise ValueError('Run a matching successful smoke probe before pilot/profile')
    if c['mode']=='gpu' and c['role']!='s1':
        from rosa_repro.vlm_contract import prepare_request
        for sample in samples:prepare_request(dict(observation=sample['observation'],**c['payload_overlay']),c['binding']['kwargs']['config'])
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_probe
    # Ray exposes vendored package metadata on import. Capture after imports so
    # the first invocation and a same-process resume have identical evidence.
    matrix=dict(suite,driver_environment=environment_evidence(),smoke_evidence=smoke)
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    # Ray caches auth state within a Python process; preserve the same private
    # token across its local-cluster restarts. Never write the token to evidence.
    if c['mode']=='cpu_dummy':os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32))
    results=[];initialized=False
    with ExperimentStore(root,matrix) as store:
        try:
            pending=[]
            for case in suite['cases']:
                if case_id is not None and case['case_id']!=case_id:continue
                if store.completed(case['case_id']):
                    number=store.db.execute("SELECT MAX(number) FROM attempts WHERE case_id=? AND status='complete'",(case['case_id'],)).fetchone()[0]
                    old=load_json(Path(root)/case['case_id']/f'attempt-{number:03d}'/'result.json')
                    if old['status']=='complete' and old['metrics']['successful']==old['metrics']['arrivals']:continue
                pending.append(case)
            for case in pending[:max_new]:
                if not initialized:
                    if c['mode']=='cpu_dummy':
                        ray.init(address='local',num_cpus=3,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',
                            _temp_dir=tempfile.mkdtemp(prefix='rosa-probe-',dir='/tmp'),log_to_driver=False)
                    else:ray.init(address=address,log_to_driver=False)
                    initialized=True;serve.start(proxy_location='Disabled')
                plan=probe_plan(c);transport=None
                with store.attempt(case['case_id']) as directory:
                    try:
                        transport=deploy_probe(plan)
                        result=await run_probe(plan,transport,samples,case,directory/'raw',corpus_id=corpus['corpus_id'],split=suite['split'])
                        write_json(directory/'result.json',result)
                        results.append(dict(case_id=case['case_id'],directory=str(directory),status=result['status'],
                            all_arrivals_successful=result.get('metrics',{}).get('arrivals',0)==result.get('metrics',{}).get('successful',-1)))
                        print(json.dumps(results[-1]),flush=True)
                    finally:
                        if transport:transport.close()
                if result['status']!='complete' or result['metrics']['successful']!=result['metrics']['arrivals']:break
        finally:
            if initialized:
                if c['mode']=='cpu_dummy':serve.shutdown()
                ray.shutdown()  # disconnect only from a GPU cluster; do not stop it
    return dict(results=results,ray_driver_disconnected=not ray.is_initialized(),pod_operations_performed=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('template');t.add_argument('--role',choices=['s1','s2','safety','monitor'],required=True)
    t.add_argument('--engine',choices=['groot','transformers','vllm'],required=True);t.add_argument('--output',type=Path,required=True)
    i=sub.add_parser('index-observations');i.add_argument('--root',type=Path,required=True);i.add_argument('--output',type=Path,required=True)
    f=sub.add_parser('preflight');f.add_argument('--config',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('prepare');s.add_argument('--config',type=Path,required=True);s.add_argument('--corpus',type=Path,required=True)
    s.add_argument('--stage',choices=['smoke','pilot','profile'],required=True);s.add_argument('--phase',choices=['exploratory','formal','cpu_smoke'],default='exploratory')
    s.add_argument('--rates',type=float,nargs='+',default=[]);s.add_argument('--repetitions',type=int,default=1);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--timing-policy',type=Path);s.add_argument('--budget-hours',type=float)
    r=sub.add_parser('run');r.add_argument('--suite',type=Path,required=True);r.add_argument('--corpus-root',type=Path,required=True)
    r.add_argument('--address');r.add_argument('--output',type=Path,required=True);r.add_argument('--max-new',type=int,default=1)
    r.add_argument('--smoke-evidence',type=Path)
    r.add_argument('--case-id',help='Execute one declared case without changing the source-bound suite')
    a=sub.add_parser('reaudit');a.add_argument('directory',type=Path);a.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('profile-candidate');e.add_argument('directories',type=Path,nargs='+');e.add_argument('--output',type=Path,required=True)
    h=sub.add_parser('pilot-rates');h.add_argument('directory',type=Path);h.add_argument('--output',type=Path,required=True)
    comp=sub.add_parser('compare');comp.add_argument('left',type=Path);comp.add_argument('right',type=Path);comp.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='template':write_json(args.output,template(args.role,args.engine))
    elif args.command=='index-observations':write_json(args.output,build_corpus(args.root))
    elif args.command=='preflight':write_json(args.output,preflight(load_json(args.config)))
    elif args.command=='prepare':write_json(args.output,make_suite(load_json(args.config),load_json(args.corpus),stage=args.stage,
        phase=args.phase,rates=args.rates,repetitions=args.repetitions,timing_policy=load_json(args.timing_policy) if args.timing_policy else None,
        budget_limit_s=args.budget_hours*3600 if args.budget_hours is not None else None))
    elif args.command=='reaudit':write_json(args.output,reaudit(args.directory))
    elif args.command=='profile-candidate':write_json(args.output,profile_candidate(args.directories))
    elif args.command=='pilot-rates':write_json(args.output,pilot_rate_hints(args.directory))
    elif args.command=='compare':write_json(args.output,compare_probes(args.left,args.right))
    else:
        result=asyncio.run(execute(load_json(args.suite),args.output,args.corpus_root,address=args.address,
            max_new=args.max_new,smoke_evidence=args.smoke_evidence,case_id=args.case_id))
        print(json.dumps(result,indent=2))
        if any(r['status']!='complete' or not r['all_arrivals_successful'] for r in result['results']):return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
