"""Actual local Ray Serve integration, using a labelled synthetic CPU policy."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import os
import secrets
import sys
import time
import uuid

os.environ['RAY_USAGE_STATS_ENABLED']='0'
os.environ['RAY_AUTH_TOKEN']=secrets.token_hex(32)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ray
from ray import serve
from rosa_repro.adapter import Request
from rosa_repro.dummy import observation
from rosa_repro.ray_service import SharedModel
from rosa_repro.regression import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run='ray-cpu-'+uuid.uuid4().hex; evidence={'status':'failed','ray_version':ray.__version__,'execution_kind':'cpu_dummy','gpu_execution':False}
    try:
        ray.init(num_cpus=4,include_dashboard=False,_node_ip_address='127.0.0.1',
                 _temp_dir='/private/tmp/rosa-ray-'+str(os.getpid()),log_to_driver=False)
        serve.start(proxy_location='Disabled')
        handle=serve.run(SharedModel.bind(max_batch_size=4,batch_wait_s=.1,dummy_delay_s=.15),name='rosa-cpu',route_prefix=None)
        requests=[]
        for i in range(12):
            ack=handle.reset.remote(run,f'r{i}','e0').result(timeout_s=10)
            requests.append(Request(run,f'r{i}','e0',f'q{i}','a','s1',0,ack['generation']))
        futures=[handle.predict.remote(asdict(r),observation(i+1)) for i,r in enumerate(requests)]
        responses=[f.result(timeout_s=20) for f in futures]
        for i,(req,res) in enumerate(zip(requests,responses)):
            assert res['metadata']==asdict(req) and res['status']=='ok'
            assert (res['action']['action.x']==i+1).all()
            assert 1<=res['actual_batch_size']<=4
        batch_stats=handle.stats.remote().result(timeout_s=10)
        assert batch_stats['model_calls']<12 and max(batch_stats['actual_batch_sizes'])>1
        assert sum(batch_stats['actual_batch_sizes'])==12 and batch_stats['max_batches_inflight']==1
        try:handle.predict.remote(asdict(requests[0]),observation(1)).result(timeout_s=10)
        except Exception: duplicate_rejected=True
        else:duplicate_rejected=False
        assert duplicate_rejected
        req=replace(requests[0],request_id='partial',observation_seq=1)
        partial=handle.predict.remote(asdict(req),observation(1)).result(timeout_s=10)
        assert partial['actual_batch_size']==1
        req=replace(req,request_id='reset-inflight',observation_seq=2)
        future=handle.predict.remote(asdict(req),observation(1))
        deadline=time.monotonic()+10
        while handle.stats.remote().result(timeout_s=10)['inflight_batches']==0:
            if time.monotonic()>deadline:raise TimeoutError('Did not observe batch in flight')
            time.sleep(.01)
        handle.reset.remote(run,'r0','e1').result(timeout_s=10)
        cancelled=future.result(timeout_s=10)
        assert cancelled['status']=='cancelled' and cancelled['action'] is None
        other=replace(requests[1],request_id='other-still-live',observation_seq=1)
        assert handle.predict.remote(asdict(other),observation(2)).result(timeout_s=10)['status']=='ok'
        final=handle.stats.remote().result(timeout_s=10)
        assert final['busy_sessions']==0
        evidence.update(status='passed',checks=dict(tensor_batching=True,response_identity=True,
            value_routing=True,duplicate_rejected=True,partial_batch=True,reset_during_inference=True,
            other_robot_isolation=True,single_model_call_at_a_time=True),first_wave=batch_stats,final=final)
    finally:
        serve.shutdown();ray.shutdown()
        write_json(a.output/'verification.json',evidence)
    print(evidence)


if __name__=='__main__':main()
