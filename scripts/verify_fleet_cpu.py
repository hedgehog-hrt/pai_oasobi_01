"""Actual four-worker Ray CPU test: routing, batching, reset/timeout and P1-P4 audit."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['RAY_USAGE_STATS_ENABLED']='0'
os.environ['RAY_AUTH_TOKEN']=secrets.token_hex(32)
import ray
from ray import serve
from rosa_repro.contracts import load_json
from rosa_repro.dummy import observation
from rosa_repro.execution_plan import validate_execution_plan,runtime_source_manifest,check_runtime_inventory
from rosa_repro.fleet_service import deploy_fleet
from rosa_repro.pipeline_runtime import run_pipeline,reaudit
from rosa_repro.regression import write_json


async def probe(plan,t):
    spec=next(w for w in plan['workers'] if w['role']=='s1');wid=spec['worker_id'];robots=spec['members']
    if len(robots)!=4 or spec['batch_size']!=4:raise ValueError('Probe requires four robots sharing a batch-four S1 worker')
    run='probe-'+uuid.uuid4().hex;acks={r:await t.reset(wid,run,r,'e0') for r in robots}
    def meta(r,q,seq):return dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run,robot_id=r,episode_id=acks[r]['episode_id'],
        request_id=q,attempt_id=q,component_id=spec['component_id'],observation_seq=seq,generation=acks[r]['generation'])
    requests=[meta(r,'q'+str(i),0) for i,r in enumerate(robots)]
    outputs=await asyncio.gather(*(t.predict(wid,m,dict(observation=observation(100+i))) for i,m in enumerate(requests)))
    assert [float(r['output']['action.x'][0,0,0]) for r in outputs]==[100.,101.,102.,103.]
    assert len({r['batch_id'] for r in outputs})==1 and all(r['actual_batch_size']==4 for r in outputs)
    partial=await t.predict(wid,meta(robots[0],'partial',1),dict(observation=observation(100)))
    assert partial['actual_batch_size']==1
    async def rejected(m):
        try:await t.predict(wid,m,dict(observation=observation(1)))
        except Exception:return True
        return False
    assert await rejected(requests[0])
    wrong=meta(robots[1],'wrong',1);wrong['component_id']='monitor';assert await rejected(wrong)
    task=asyncio.create_task(t.predict(wid,meta(robots[0],'reset-flight',2),dict(observation=observation(100))))
    deadline=time.monotonic()+5
    # Observe only the target replica; waiting for every worker can miss this
    # short 80 ms synthetic model call under unrelated host/RPC load.
    while (await t.handles[wid].stats.remote())['inflight_batches']==0:
        if task.done():raise AssertionError('Reset probe completed before its in-flight state was observed')
        if time.monotonic()>deadline:raise TimeoutError('Never observed model in flight')
        await asyncio.sleep(.002)
    acks[robots[0]]=await t.reset(wid,run,robots[0],'e1')
    cancelled=await task;assert cancelled['status']=='cancelled' and cancelled['output'] is None
    other=await t.predict(wid,meta(robots[1],'other',1),dict(observation=observation(101)))
    assert other['status']=='ok'
    try:await asyncio.wait_for(t.predict(wid,meta(robots[0],'timeout',0),dict(observation=observation(100))),.01)
    except TimeoutError:pass
    else:raise AssertionError('Expected injected short client timeout')
    busy=await t.predict(wid,meta(robots[0],'busy',1),dict(observation=observation(100)))
    assert busy['status']=='rejected'
    while (await t.inventory())[wid]['busy_sessions']:
        if time.monotonic()>deadline+5:raise TimeoutError('Late work did not drain')
        await asyncio.sleep(.005)
    recovered=await t.predict(wid,meta(robots[0],'recovered',2),dict(observation=observation(100)))
    assert recovered['status']=='ok'
    return dict(tensor_batch_four=True,value_and_response_routing=True,partial_batch=True,duplicate_rejected=True,
                wrong_component_rejected=True,reset_during_model_call=True,other_robot_isolated=True,
                timeout_does_not_free_busy_session=True,recovery_after_late_completion=True)


async def exercise(plan,t,out):
    checks=await probe(plan,t);write_json(out/'protocol_checks.json',checks)
    robots=[r['robot_id'] for c in plan['classes'] for r in c['robots']]
    result=await run_pipeline(plan,t,lambda r,seq:observation(robots.index(r)+1),out/'pipeline',
        warmup_s=2.1,duration_s=4.2,request_timeout_s=10,source_provenance=dict(kind='synthetic',robot_values=dict(zip(robots,range(1,5)))))
    metrics=reaudit(out/'pipeline',out/'recomputed_metrics.json')
    assert metrics==load_json(out/'pipeline/metrics.json')
    assert result['status']=='complete' and all(c['qualified_action_chunks']>0 for c in metrics['classes'].values())
    before=load_json(out/'pipeline/workers_before.json');after=load_json(out/'pipeline/workers_after.json')
    check_runtime_inventory(plan,after)
    assert len({(x['node_id'],x['pid']) for x in after.values()})==4
    s1=next(w['worker_id'] for w in plan['workers'] if w['role']=='s1')
    s2=next(w['worker_id'] for w in plan['workers'] if w['role']=='s2')
    batch_sizes=after[s1]['actual_batch_sizes'][len(before[s1]['actual_batch_sizes']):]
    assert max(batch_sizes)>1 and after[s2]['model_calls']-before[s2]['model_calls']>=2
    events=load_json(out/'pipeline/audit.json')['records']
    for c in plan['classes']:
        for robot in c['robots']:
            rs=sorted((r for r in events if r['record_type']=='request' and r['robot_id']==robot['robot_id'] and r['component_id']=='s1'),key=lambda x:x['sent_ns'])
            assert all(b['sent_ns']-a['sent_ns']>=1e9/c['frequency_hz'] for a,b in zip(rs,rs[1:]))
    return dict(protocol_checks=checks,driver_result=result,qualified_chunks_by_class={k:v['qualified_action_chunks'] for k,v in metrics['classes'].items()},
        s1_actual_batch_sizes=batch_sizes,s1_requests=sum(batch_sizes),s1_model_calls=len(batch_sizes),
        s2_model_calls=after[s2]['model_calls']-before[s2]['model_calls'],distinct_worker_processes=4,
        frequency_caps_verified=True,raw_log_reaudit_identical=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=validate_execution_plan(load_json(a.plan))
    if plan['mode']!='cpu_dummy':p.error('This verifier never starts GPU workloads')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='failed',ray_version=ray.__version__,gpu_execution=False,source_manifest=runtime_source_manifest(),
                plan_file_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),ray_shutdown_completed=False)
    transport=None
    try:
        ray.init(address='local',num_cpus=6,num_gpus=0,include_dashboard=False,_node_ip_address='127.0.0.1',
            _temp_dir=tempfile.mkdtemp(prefix='rosa-fleet-verify-',dir='/private/tmp'),log_to_driver=False)
        serve.start(proxy_location='Disabled');transport=deploy_fleet(plan)
        report.update(asyncio.run(exercise(plan,transport,a.output)),status='passed')
    finally:
        try:
            if transport:transport.close()
        finally:
            if ray.is_initialized():serve.shutdown();ray.shutdown()
            report['ray_shutdown_completed']=not ray.is_initialized()
            write_json(a.output/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_manifest'},indent=2))


if __name__=='__main__':main()
