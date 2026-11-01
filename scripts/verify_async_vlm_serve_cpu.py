"""Actual Ray asynchronous admission test using explicitly synthetic VLM outputs."""
import argparse,asyncio,json,os,secrets,sys,tempfile,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['RAY_USAGE_STATS_ENABLED']='0';os.environ['RAY_AUTH_TOKEN']=secrets.token_hex(32)
import ray
from ray import serve
from rosa_repro.component_runtime import ComponentCore
from rosa_repro.execution_plan import compile_execution_plan,runtime_source_manifest
from rosa_repro.fleet_service import deploy_fleet
from rosa_repro.contracts import load_json
from rosa_repro.dummy import observation
from rosa_repro.regression import write_json


async def exercise(plan,transport):
    spec=next(w for w in plan['workers'] if w['role']=='monitor');wid=spec['worker_id']
    run=uuid.uuid4().hex;requests=[]
    for i,rid in enumerate(spec['members']):
        ack=await transport.reset(wid,run,rid,'e')
        requests.append(dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run,robot_id=rid,episode_id='e',
            request_id=f'q{i}',attempt_id=f'a{i}',component_id='monitor',observation_seq=0,generation=ack['generation']))
    assert len(requests)==3
    responses=await asyncio.gather(*(transport.predict(wid,m,{'observation':observation(i)}) for i,m in enumerate(requests)))
    assert all(r['status']=='ok' and r['output']['task_status']=='ongoing' and r['actual_batch_size']==1 for r in responses)
    assert [r['metadata'] for r in responses]==requests
    assert all(r['backend_evidence']['evidence_kind']=='synthetic_async_probe' for r in responses)
    stats=(await transport.inventory())[wid]
    assert stats['max_inflight_batches']==2 and stats['configured_concurrency']==2 and stats['busy_sessions']==0
    assert stats['python_executable']==spec['backend']['runtime_env']['py_executable']
    return dict(requests=3,concurrent_backend_invocations_observed=2,backend_evidence_transferred=True,
                response_identity_preserved=True,explicit_python_path_verified=True,admission_batch_sizes=[r['actual_batch_size'] for r in responses],
                internal_vllm_token_batch_sizes_measured=False,worker=stats)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x=load_json(a.plan)['inputs'];x['bindings']['monitor']['kwargs']=dict(delay_s=.08,async_concurrency=2)
    x['bindings']['monitor']['runtime_env']=dict(py_executable=sys.executable)
    plan=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='cpu_dummy',bindings=x['bindings'])
    a.output.mkdir(parents=True,exist_ok=False);write_json(a.output/'execution_plan.json',plan)
    report=dict(status='failed',gpu_execution=False,actual_vllm_engine_loaded=False,source_manifest=runtime_source_manifest(),ray_version=ray.__version__)
    transport=None
    try:
        ray.init(address='local',num_cpus=6,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',
                 _temp_dir=tempfile.mkdtemp(prefix='rosa-async-',dir='/private/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy_fleet(plan)
        report.update(asyncio.run(exercise(plan,transport)),status='passed')
    finally:
        try:
            if transport:transport.close()
        finally:
            if ray.is_initialized():serve.shutdown();ray.shutdown()
            report['ray_shutdown_completed']=not ray.is_initialized();write_json(a.output/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_manifest','worker')},indent=2))


if __name__=='__main__':main()
