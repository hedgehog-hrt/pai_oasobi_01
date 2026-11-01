"""Prepare/capture real LIBERO observations and evaluate S2/Safety/Monitor semantics."""
import argparse,asyncio,json,os,secrets,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.capture_plan import default_design,make_capture_plan,capture,validate_capture_plan
from rosa_repro.contracts import load_json
from rosa_repro.libero_bridge import T01_NAME,export_t01_bank,LiberoEpisode,ZmqRegressionTransport
from rosa_repro.observation_quality import (default_criteria,prepare_quality,validate_quality_suite,annotation_template,
    rubric_template,evaluate_quality,export_review_images)
from rosa_repro.profile_protocol import probe_plan,load_corpus
from rosa_repro.quality_runner import run_quality
from rosa_repro.regression import write_json,verify_upstream


async def execute_quality(suite,root,output,address=None):
    validate_quality_suite(suite,root);c=suite['config']
    if Path(output).exists():raise FileExistsError(output)
    if c['mode']=='gpu' and (not address or address=='local'):raise ValueError('GPU run requires an explicit existing Ray address')
    if c['mode']=='cpu_dummy' and address:raise ValueError('CPU harness cannot attach to an external cluster')
    if c['mode']=='gpu':
        from rosa_repro.vlm_contract import prepare_request
        _,samples=load_corpus(suite['corpus'],root,suite['split'],c['mode'])
        for x in samples:prepare_request(dict(observation=x['observation'],**c['payload_overlay']),c['binding']['kwargs']['config'])
    import ray
    from ray import serve
    from rosa_repro.fleet_service import deploy_probe
    os.environ['RAY_USAGE_STATS_ENABLED']='0';transport=None
    if c['mode']=='cpu_dummy':os.environ.setdefault('RAY_AUTH_TOKEN',secrets.token_hex(32))
    try:
        if c['mode']=='cpu_dummy':
            ray.init(address='local',num_cpus=3,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',
                _temp_dir=tempfile.mkdtemp(prefix='rosa-quality-',dir='/tmp'),log_to_driver=False)
        else:ray.init(address=address,log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy_probe(probe_plan(c))
        return await run_quality(suite,root,transport,output)
    finally:
        if transport:transport.close()
        if ray.is_initialized():
            if c['mode']=='cpu_dummy':serve.shutdown()
            ray.shutdown()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('design','criteria'):
        q=sub.add_parser(name);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('export-states');q.add_argument('--repo',type=Path,required=True);q.add_argument('--task',default=T01_NAME)
    q.add_argument('--count',type=int,default=30);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('plan');q.add_argument('--bank',type=Path,nargs='+',required=True);q.add_argument('--design',type=Path)
    q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('capture');q.add_argument('--repo',type=Path,required=True);q.add_argument('--plan',type=Path,required=True)
    q.add_argument('--bank',type=Path,nargs='+',required=True);q.add_argument('--port',type=int,default=5556);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('prepare-quality');q.add_argument('--config',type=Path,required=True);q.add_argument('--corpus',type=Path,required=True)
    q.add_argument('--root',type=Path,required=True);q.add_argument('--split',choices=['evaluation','holdout'],default='holdout')
    q.add_argument('--criteria',type=Path);q.add_argument('--output',type=Path,required=True)
    for name in ('annotations','review-images','run-quality','rubric','evaluate'):
        q=sub.add_parser(name);q.add_argument('--suite',type=Path,required=True);q.add_argument('--root',type=Path,required=True)
        q.add_argument('--output',type=Path,required=True)
        if name=='run-quality':q.add_argument('--address')
        if name in ('rubric','evaluate'):q.add_argument('--predictions',type=Path,required=True)
        if name=='evaluate':q.add_argument('--references',type=Path);q.add_argument('--ratings',type=Path)
    a=p.parse_args(argv)
    if a.command=='design':write_json(a.output,default_design())
    elif a.command=='criteria':write_json(a.output,default_criteria())
    elif a.command=='plan':write_json(a.output,make_capture_plan(a.bank,load_json(a.design) if a.design else None))
    elif a.command in ('export-states','capture'):
        if a.command=='capture':plan=validate_capture_plan(load_json(a.plan),a.bank)
        verify_upstream(a.repo);sys.path.insert(0,str(a.repo.resolve()))
        if a.command=='export-states':export_t01_bank(a.output,a.count,task_name=a.task)
        else:
            from gr00t.policy.server_client import PolicyClient
            with PolicyClient(host='127.0.0.1',port=a.port) as client:configs=client.get_modality_config()
            d=plan['design'];result=capture(plan,a.bank,a.output,
                lambda task:LiberoEpisode(configs,d['max_steps'],d['execute_steps'],task_name=task),
                lambda:ZmqRegressionTransport('official',a.port),execution_kind='libero_gpu')
            return 0 if result['completed_episodes']==result['intended_episodes'] else 1
    elif a.command=='prepare-quality':write_json(a.output,prepare_quality(load_json(a.config),load_json(a.corpus),a.root,
        split=a.split,criteria=load_json(a.criteria) if a.criteria else None))
    else:
        s=validate_quality_suite(load_json(a.suite),a.root,current_source=a.command=='run-quality')
        if a.command=='annotations':write_json(a.output,annotation_template(s))
        elif a.command=='review-images':export_review_images(s,a.root,a.output)
        elif a.command=='run-quality':
            result=asyncio.run(execute_quality(s,a.root,a.output,a.address));print(json.dumps({k:v for k,v in result.items() if k in ('status','drained','suite_id')}))
            return 0 if result['status']=='complete' and all(r['status']=='ok' for r in result['rows']) else 1
        elif a.command=='rubric':write_json(a.output,rubric_template(s,load_json(a.predictions)))
        else:
            result=evaluate_quality(s,load_json(a.predictions),load_json(a.references) if a.references else None,load_json(a.ratings) if a.ratings else None)
            write_json(a.output,result);return 0 if result['status']=='passed' else 2
    return 0


if __name__=='__main__':raise SystemExit(main())
