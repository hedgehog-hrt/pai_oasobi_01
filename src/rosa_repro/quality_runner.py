"""One request per saved observation on the actual component transport; no actions."""
import asyncio,time,uuid
from copy import deepcopy
from pathlib import Path
from .adapter import JsonlSink
from .component_runtime import validate_output
from .execution_plan import check_runtime_inventory,runtime_source_manifest
from .observation_quality import validate_quality_suite,seal_predictions
from .paired import digest_value
from .profile_protocol import load_corpus,probe_plan
from .profile_runner import seal
from .regression import write_json


async def run_quality(suite,root,transport,output):
    validate_quality_suite(suite,root);c=suite['config'];plan=probe_plan(c);wid=plan['workers'][0]['worker_id']
    _,samples=load_corpus(suite['corpus'],root,suite['split'],c['mode'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'condition.json',suite);write_json(out/'source_manifest.json',runtime_source_manifest())
    rows=[];run_id='quality-'+uuid.uuid4().hex;sink=JsonlSink(out/'requests.jsonl');pending=None
    summary=dict(suite_id=suite['suite_id'],mode=c['mode'],status='failed',drained=False,actions_executed=0,
        gpu_execution_requested=c['mode']=='gpu',performance_benchmark=False,source_manifest=runtime_source_manifest())
    def record(row):rows.append(row);sink(row)
    async def bounded(value):return await asyncio.wait_for(value,c['request_timeout_s'])
    try:
        before=await bounded(transport.inventory());check_runtime_inventory(plan,before);summary['worker_before']=before[wid]
        summary['worker_environment']=await bounded(transport.environment(wid));instance=before[wid]['instance_id']
        for i,sample in enumerate(samples):
            rid=plan['workers'][0]['members'][0];episode='sample-'+str(i)
            ack=await bounded(transport.reset(wid,run_id,rid,episode))
            meta=dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run_id,robot_id=rid,episode_id=episode,
                request_id=sample['sample_id'],attempt_id=sample['sample_id'],component_id=c['component_id'],
                observation_seq=0,generation=ack['generation'])
            payload=dict(observation=deepcopy(sample['observation']),**deepcopy(c['payload_overlay']))
            row=dict(sample_id=sample['sample_id'],observation_sha256=sample['sha256'],metadata=meta,
                input_sha256=digest_value(payload),sent_ns=time.monotonic_ns(),status='error')
            abort=False
            try:
                pending=asyncio.create_task(transport.predict(wid,meta,payload))
                response=await asyncio.wait_for(asyncio.shield(pending),c['request_timeout_s'])
                row['finished_ns']=time.monotonic_ns()
                if response['metadata']!=meta or response['instance_id']!=instance or response['worker_id']!=wid or response['plan_id']!=plan['plan_id']:
                    raise ValueError('Response identity changed')
                row.update(status=response['status'],output=response.get('output'),backend_evidence=response.get('backend_evidence'),
                    error_type=response.get('error_type'),actual_batch_size=response.get('actual_batch_size'))
                if row['status'] not in ('ok','error','timeout','rejected','cancelled'):raise ValueError('Unknown status')
                if row['status']=='ok':validate_output(c['role'],row['output'])
                # A worker-side timeout may still own the backend. Do not send the next observation.
                abort=row['status'] in ('timeout','rejected','cancelled')
            except TimeoutError:row.update(status='timeout',error_type='ClientTimeout');abort=True
            except Exception as exc:row.update(status='error',error_type=type(exc).__name__);abort=True
            finally:
                row.setdefault('finished_ns',time.monotonic_ns());record(row)
            if abort:break
        if pending is not None:
            try:await asyncio.wait_for(asyncio.shield(pending),c['drain_timeout_s'])
            except TimeoutError:raise TimeoutError('Quality backend did not drain')
        after=await bounded(transport.inventory());check_runtime_inventory(plan,after)
        summary['worker_after']=after[wid]
        summary['drained']=after[wid]['instance_id']==instance and after[wid]['busy_sessions']==after[wid]['inflight_batches']==0
        summary['status']='complete' if len(rows)==len(samples) and summary['drained'] else 'partial'
    except Exception as exc:summary['error']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if pending is not None:
            if not pending.done():pending.cancel()
            await asyncio.gather(pending,return_exceptions=True)
        finished={r['sample_id'] for r in rows}
        for sample in samples:
            if sample['sample_id'] not in finished:record(dict(sample_id=sample['sample_id'],status='not_run'))
        sink.close();bundle=seal_predictions(dict(summary,rows=rows))
        write_json(out/'predictions.json',bundle);write_json(out/'result.json',summary);seal(out)
    return bundle
