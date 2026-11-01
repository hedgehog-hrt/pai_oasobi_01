"""Compile R07 -> Ray bindings, run bounded virtual replay, or re-audit raw logs."""
import argparse
import asyncio
from pathlib import Path
import os
import secrets
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.contracts import load_json
from rosa_repro.execution_plan import compile_execution_plan,validate_execution_plan
from rosa_repro.regression import write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('compile');c.add_argument('--fleet',required=True);c.add_argument('--schedule',required=True)
    c.add_argument('--roles',required=True);c.add_argument('--bindings');c.add_argument('--mode',choices=['cpu_dummy','gpu'],default='cpu_dummy')
    c.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--plan',required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--warmup',type=float,default=2.1);r.add_argument('--duration',type=float,default=5.)
    r.add_argument('--timeout',type=float,default=5.);r.add_argument('--address');r.add_argument('--observation-manifest')
    r.add_argument('--adoptions',type=Path,help='Reviewed profile mapping and evidence_root; required for GPU R07')
    a=sub.add_parser('reaudit');a.add_argument('directory');a.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='compile':
        plan=compile_execution_plan(load_json(args.fleet),load_json(args.schedule),load_json(args.roles),mode=args.mode,
                                    bindings=load_json(args.bindings) if args.bindings else None)
        write_json(args.output,plan);print('Saved execution plan:',args.output);return 0
    from rosa_repro.pipeline_runtime import run_pipeline,reaudit
    if args.command=='reaudit':reaudit(args.directory,args.output);return 0
    plan=validate_execution_plan(load_json(args.plan))
    from rosa_repro.profile_adoption import validate_fleet_adoptions
    adoptions=load_json(args.adoptions) if args.adoptions else None
    validate_fleet_adoptions(plan,adoptions)
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    from rosa_repro.dummy import observation
    from rosa_repro.paired import digest_value
    if plan['mode']=='gpu':
        if not args.address or not args.observation_manifest:p.error('GPU run requires explicit existing Ray address and saved-observation manifest')
        from rosa_repro.regression import load_observation
        manifest=load_json(args.observation_manifest);observations=[]
        for row in manifest['observations']:
            obs,_=load_observation(row['directory'])
            if digest_value(obs)!=row['sha256']:raise ValueError('Saved observation content hash mismatch')
            observations.append(obs)
        if not observations:raise ValueError('Saved observations are empty')
        source=lambda robot,seq:observations[seq%len(observations)]
        provenance=dict(kind='saved_observation_replay',manifest=manifest)
    else:
        if args.address or args.observation_manifest:p.error('CPU dummy run uses a new local cluster and synthetic observations')
        robots=[r['robot_id'] for c in plan['classes'] for r in c['robots']]
        source=lambda robot,seq:observation(robots.index(robot)+1)
        provenance=dict(kind='synthetic',robots=robots)
    os.environ['RAY_USAGE_STATS_ENABLED']='0'
    if plan['mode']=='cpu_dummy':os.environ['RAY_AUTH_TOKEN']=secrets.token_hex(32)
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_fleet
    transport=None
    try:
        if plan['mode']=='cpu_dummy':
            ray.init(address='local',num_cpus=len(plan['workers'])+2,num_gpus=0,include_dashboard=False,
                _node_ip_address='127.0.0.1',_temp_dir=tempfile.mkdtemp(prefix='rosa-fleet-',dir='/private/tmp'),log_to_driver=False)
        else:ray.init(address=args.address,log_to_driver=False)
        serve.start(proxy_location='Disabled')
        transport=deploy_fleet(plan,adoptions=adoptions)
        result=asyncio.run(run_pipeline(plan,transport,source,args.output,warmup_s=args.warmup,duration_s=args.duration,
                                       request_timeout_s=args.timeout,source_provenance=provenance))
        print('Run status:',result['status'],'; raw logs:',args.output)
        return 0 if result['status']=='complete' and result.get('drained') else 1
    finally:
        try:
            if transport:transport.close()
        finally:
            if ray.is_initialized():
                if plan['mode']=='cpu_dummy':serve.shutdown()
                ray.shutdown()


if __name__=='__main__':raise SystemExit(main())
