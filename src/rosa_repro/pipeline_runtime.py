"""P1-P4 driver: periodic obligations, S2 reuse, rate caps and durable v2 logs.
Single driver monotonic clock; server clocks never enter the evaluator.
"""
import asyncio
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import uuid

from .adapter import JsonlSink,ProtocolError
from .component_runtime import validate_output
from .execution_plan import validate_execution_plan,check_runtime_inventory,runtime_source_manifest
from .multicomponent import evaluate_multicomponent
from .paired import digest_value
from .regression import write_json


async def sleep_until(target_ns):
    while time.monotonic_ns()<target_ns:await asyncio.sleep((target_ns-time.monotonic_ns())/1e9)


async def virtual_execute(robot_id,action,action_time_s):
    await asyncio.sleep(action_time_s)
    return 1  # A virtual timed chunk, never a claim of robot/simulator steps.


async def run_pipeline(plan,transport,observation_source,output,*,warmup_s=2.1,duration_s=5.,
                       request_timeout_s=5.,max_requests=100000,execute=None,
                       action_semantics='virtual_timed_chunks',source_provenance=None):
    validate_execution_plan(plan)
    if any(not math.isfinite(v) or v<0 for v in (warmup_s,duration_s,request_timeout_s)) or duration_s<=0 or request_timeout_s<=0:
        raise ValueError('Invalid bounded run duration/timeout')
    if type(max_requests) is not int or not 1<=max_requests<=100000:raise ValueError('Invalid request cap')
    if execute is None:
        if action_semantics!='virtual_timed_chunks':raise ValueError('External action semantics require an executor')
        execute=virtual_execute
    elif action_semantics!='external_executor_reported_steps':raise ValueError('External executor semantics must be declared')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'execution_plan.json',plan)
    write_json(output/'source_manifest.json',runtime_source_manifest())
    write_json(output/'protocol.json',dict(warmup_s=warmup_s,duration_s=duration_s,request_timeout_s=request_timeout_s,
        max_requests=max_requests,action_semantics=action_semantics,source_provenance=source_provenance,
        measurement_phase='connection_validation',formal_performance_claim=False))
    events=JsonlSink(output/'events.jsonl');details=JsonlSink(output/'request_details.jsonl');lifecycle=JsonlSink(output/'lifecycle.jsonl')
    records=[];errors=[];sessions={};states={};workers={w['worker_id']:w for w in plan['workers']}
    run_id='pipeline-'+uuid.uuid4().hex;domain=run_id+'-driver-monotonic';request_count=0
    result=dict(status='failed',run_id=run_id,plan_id=plan['plan_id'],execution_mode=plan['mode'],
                action_semantics=action_semantics,gpu_kernel_execution_verified=False,formal_performance_claim=False)
    epoch=None;tasks=[];audit=None;fatal=None
    def emit(row):events(row);records.append(row)
    async def inventory():
        return await asyncio.wait_for(transport.inventory(),request_timeout_s)
    try:
        before=await inventory();check_runtime_inventory(plan,before);write_json(output/'workers_before.json',before)
        for c in plan['classes']:
            for robot in c['robots']:
                rid=robot['robot_id'];states[rid]=dict(plan=None,uses=0,safety=None,safety_expiry_ns=0,
                    completed_chunks=0,reported_steps=0,last_action_request_id=None)
                for comp,wid in robot['routes'].items():
                    ack=await asyncio.wait_for(transport.reset(wid,run_id,rid,'episode0'),request_timeout_s)
                    expected=dict(run_id=run_id,robot_id=rid,episode_id='episode0',worker_id=wid,plan_id=plan['plan_id'])
                    if any(ack.get(k)!=v for k,v in expected.items()) or ack.get('instance_id')!=before[wid]['instance_id'] or type(ack.get('generation')) is not int:
                        raise ProtocolError('Reset acknowledgement identity mismatch')
                    sessions[rid,comp]=dict(ack=ack,seq=0)
        epoch=time.monotonic_ns();start=epoch+round(warmup_s*1e9);end=start+round(duration_s*1e9)
        lifecycle(dict(kind='run_started',run_id=run_id,clock_domain=domain,epoch_ns=epoch,window_start_ns=start,window_end_ns=end,
                       classes=[c['audit'] for c in plan['classes']],episodes=[dict(class_id=c['audit']['id'],robot_id=r['robot_id'],
                       episode_id='episode0',start_ns=epoch) for c in plan['classes'] for r in c['robots']]))
        async def request(c,robot,comp,*,scheduled=None,slot=None,plan_ref=None,plan_text=None):
            nonlocal request_count
            if request_count>=max_requests:raise RuntimeError('Run request cap reached')
            rid=robot['robot_id'];mid=comp['id'];wid=robot['routes'][mid];session=sessions[rid,mid]
            state=states[rid]
            current_plan=plan_text if comp['role']=='s1' else state['plan'][1] if state['plan'] else None
            history=dict(action_semantics=action_semantics,completed_chunks=state['completed_chunks'],
                         reported_steps=state['reported_steps'],last_action_request_id=state['last_action_request_id'])
            payload=dict(observation=deepcopy(observation_source(rid,session['seq'])),plan=current_plan,plan_request_id=plan_ref,
                         context={'execution_history':json.dumps(history,sort_keys=True)})
            meta=dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run_id,robot_id=rid,episode_id='episode0',
                request_id=uuid.uuid4().hex,attempt_id=uuid.uuid4().hex,component_id=mid,
                observation_seq=session['seq'],generation=session['ack']['generation'])
            session['seq']+=1;request_count+=1;sent=time.monotonic_ns()
            row=dict(schema_version='2.0',record_type='request',**{k:meta[k] for k in
                ('run_id','robot_id','episode_id','request_id','attempt_id','component_id','observation_seq')},
                clock_domain=domain,scheduled_ns=sent if scheduled is None else scheduled,sent_ns=sent,finished_ns=sent,
                deadline_ms=comp['deadline_ms'],status='error',valid_output=False,warmup=sent<start,
                batch_id=None,actual_batch_size=None,slot_index=slot,plan_request_id=plan_ref)
            value=None;response=None;error=None
            # An intent survives a driver crash; a missing completion is never fabricated as success.
            lifecycle(dict(kind='request_started',metadata=meta,scheduled_ns=row['scheduled_ns'],sent_ns=sent,slot_index=slot))
            try:
                response=await asyncio.wait_for(transport.predict(wid,meta,payload),request_timeout_s)
                if response.get('metadata')!=meta or any(response.get(k)!=v for k,v in
                    dict(worker_id=wid,plan_id=plan['plan_id'],instance_id=session['ack']['instance_id']).items()):
                    raise ProtocolError('Response route/session/instance mismatch')
                status=response.get('status')
                if status not in ('ok','error','timeout','rejected','cancelled'):raise ProtocolError('Invalid response status')
                bid,size=response.get('batch_id'),response.get('actual_batch_size')
                if (bid is None)!=(size is None) or (size is not None and
                    (not isinstance(bid,str) or not bid or type(size) is not int or not 1<=size<=workers[wid]['batch_size'])):
                    raise ProtocolError('Invalid batch evidence')
                row.update(status=status,batch_id=bid,actual_batch_size=size)
                if status=='ok':
                    if size is None:raise ProtocolError('Successful output lacks batch evidence')
                    validate_output(comp['role'],response['output']);value=response['output'];row['valid_output']=True
            except asyncio.CancelledError:
                row['status']='cancelled';raise
            except TimeoutError:row['status']='timeout';error='TimeoutError'
            except ProtocolError:
                row.update(status='error',valid_output=False);raise
            except Exception as exc:row.update(status='error',valid_output=False);error=type(exc).__name__
            finally:
                row['finished_ns']=time.monotonic_ns()
                details(dict(request_id=meta['request_id'],worker_id=wid,plan_id=plan['plan_id'],
                    expected_instance_id=session['ack']['instance_id'],input_sha256=digest_value(payload),
                    output_sha256=digest_value(value) if value is not None else None,
                    semantic_output=value if value is not None and comp['role']!='s1' else None,
                    batch_id=row['batch_id'],server_timing=response.get('server_timing') if isinstance(response,dict) else None,
                    backend_evidence=response.get('backend_evidence') if isinstance(response,dict) else None,
                    error_type=error or (response.get('error_type') if isinstance(response,dict) else None)))
                emit(row)
            return row,value
        async def periodic(c,robot,comp):
            period=comp['period_ns'];base=epoch+comp['phase_ns'];index=0
            while True:
                scheduled=base+index*period
                if scheduled>=end:return
                await sleep_until(scheduled)
                now=time.monotonic_ns()
                if now>=end:return
                current=(now-base)//period
                if current>index:
                    lifecycle(dict(kind='periodic_slots_skipped',robot_id=robot['robot_id'],component_id=comp['id'],
                                   first_slot=index,last_slot=current-1,reason='no_catchup_burst'))
                    index=current;scheduled=base+index*period
                row,value=await request(c,robot,comp,scheduled=scheduled,slot=index)
                if comp['role']=='safety':
                    state=states[robot['robot_id']]
                    timely=row['finished_ns']<=scheduled+round(comp['deadline_ms']*1e6)
                    state['safety']=value['stop'] if value is not None and timely else None
                    state['safety_expiry_ns']=scheduled+period+round(comp['deadline_ms']*1e6)
                index+=1
        async def goal(c,robot):
            f=c['frequency_hz']
            if f==0:return
            interval=0 if f is None else math.ceil(1e9/f);next_send=epoch
            specs={x['role']:x for x in c['audit']['components']};rid=robot['robot_id'];state=states[rid]
            while True:
                await sleep_until(min(next_send,end))
                if time.monotonic_ns()>=end:return
                if 's2' in specs and (state['plan'] is None or state['uses']>=c['audit']['s2_horizon']):
                    p,value=await request(c,robot,specs['s2'])
                    if value is None:
                        state['plan']=None;next_send=p['sent_ns']+interval;continue
                    state['plan']=(p['request_id'],value['plan']);state['uses']=0
                if time.monotonic_ns()>=end:return
                row,action=await request(c,robot,specs['s1'],plan_ref=state['plan'][0] if state['plan'] else None,
                                         plan_text=state['plan'][1] if state['plan'] else None)
                state['uses']+=1;next_send=row['sent_ns']+interval
                if action is None or time.monotonic_ns()>=end:continue
                stopped='safety' in specs and (state['safety'] is not False or time.monotonic_ns()>=state['safety_expiry_ns'])
                begin=time.monotonic_ns();steps=0
                lifecycle(dict(kind='action_started',request_id=row['request_id'],robot_id=rid,start_ns=begin,safety_stopped=stopped))
                if not stopped:
                    steps=await asyncio.wait_for(execute(rid,deepcopy(action),c['action_time_s']),
                                                 c['action_time_s']+request_timeout_s)
                    if type(steps) is not int or not 0<=steps<=min(v.shape[1] for v in action.values()):
                        raise ProtocolError('Executor must report actual steps within action horizon')
                emit(dict(schema_version='2.0',record_type='action',**{k:row[k] for k in
                    ('run_id','clock_domain','robot_id','episode_id','request_id','component_id')},
                    action_id=uuid.uuid4().hex,start_ns=begin,end_ns=time.monotonic_ns(),applied=steps>0,
                    executed_steps=steps,safety_stopped=stopped))
                if steps:
                    state['completed_chunks']+=1;state['reported_steps']+=steps;state['last_action_request_id']=row['request_id']
        for c in plan['classes']:
            for robot in c['robots']:
                tasks.append(asyncio.create_task(goal(c,robot)))
                for comp in c['audit']['components']:
                    if comp['role'] in ('safety','monitor'):tasks.append(asyncio.create_task(periodic(c,robot,comp)))
        await asyncio.gather(*tasks)
        await sleep_until(end)
        # Late model work may outlive a transport timeout. Do not claim a clean drain.
        drain_deadline=time.monotonic()+request_timeout_s
        while True:
            after=await inventory()
            if not any(s['busy_sessions'] or s['inflight_batches'] for s in after.values()):break
            if time.monotonic()>=drain_deadline:raise TimeoutError('Worker drain incomplete')
            await asyncio.sleep(.01)
        check_runtime_inventory(plan,after)
        if any(after[w]['instance_id']!=before[w]['instance_id'] for w in before):raise ProtocolError('Worker restarted during run')
        write_json(output/'workers_after.json',after)
        finish=time.monotonic_ns();episode_end=finish+1
        audit=dict(schema_version='2.0',run_id=run_id,clock_domain=domain,run_epoch_ns=epoch,window_start_ns=start,
            window_end_ns=end,audit_end_ns=max(end,episode_end),classes=[c['audit'] for c in plan['classes']],
            episodes=[dict(class_id=c['audit']['id'],robot_id=r['robot_id'],episode_id='episode0',start_ns=epoch,end_ns=episode_end)
                      for c in plan['classes'] for r in c['robots']],records=records)
        metrics=evaluate_multicomponent(audit)
        write_json(output/'audit.json',audit);write_json(output/'metrics.json',metrics)
        statuses=Counter(r['status'] for r in records if r['record_type']=='request')
        result.update(status='complete' if set(statuses)<={'ok'} else 'complete_with_request_failures',
            request_statuses=dict(statuses),requests=request_count,qualified_action_chunks=metrics['qualified_action_chunks'],
            weighted_Q=metrics['weighted_Q'],drained=True)
        lifecycle(dict(kind='run_finished',audit_end_ns=audit['audit_end_ns'],episode_end_ns=episode_end))
    except BaseException as exc:
        fatal=exc;result.update(error_type=type(exc).__name__,error=str(exc),drained=False)
        for t in tasks:
            if not t.done():t.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
        lifecycle(dict(kind='run_failed',error_type=type(exc).__name__,completed_records=len(records)))
    finally:
        for sink in (events,details,lifecycle):sink.close()
        write_json(output/'result.json',result)
    if fatal:raise fatal
    return result


def reaudit(directory,output):
    """Recompute from persisted raw events; refuse missing/crashed completion."""
    from .contracts import load_json
    directory=Path(directory);a=load_json(directory/'audit.json')
    records=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    if a['records']!=records:raise ValueError('Raw events disagree with finalized audit snapshot')
    result=evaluate_multicomponent(a);write_json(output,result);return result
