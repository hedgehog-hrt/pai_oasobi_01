"""Open-loop profile and closed-loop virtual-action replay over a Serve handle.

All timestamps used by R00 are from this driver. Server timings are separate.
Synthetic CPU runs are never exported as measured GPU model profiles.
"""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .adapter import JsonlSink,MeasuredClient
from .measurement import evaluate_p1
from .regression import write_json


async def run_trial(handle,observations,case,output):
    if not observations:raise ValueError('At least one saved observation required')
    output=Path(output)
    write_json(output/'condition.json',case)
    run_id=case['case_id']+'-'+uuid.uuid4().hex
    domain=run_id+'-driver-monotonic'
    trace=JsonlSink(output/'events.jsonl');timings=JsonlSink(output/'server_timings.jsonl')
    records=[];guard=threading.Lock();counts=Counter();fatal=[]
    def emit(record):
        with guard:
            records.append(record)
            if record['record_type']=='request':
                counts[(record['warmup'],'all')]+=1
                if record['status']=='ok' and record['valid_output']:counts[(record['warmup'],'success')]+=1
            trace(record)
    timeout=case['protocol']['request_timeout_s']
    loop=asyncio.get_running_loop()
    pool=ThreadPoolExecutor(max_workers=case['robots']+2)
    async def blocking(fn,*args):return await loop.run_in_executor(pool,fn,*args)
    def reset(run,robot,episode):return handle.reset.remote(run,robot,episode).result(timeout_s=timeout)
    def predict(meta,obs):
        try:res=handle.predict.remote(meta,obs).result(timeout_s=timeout)
        except TimeoutError as exc:raise TimeoutError('Serve response deadline for transport exceeded') from exc
        if 'server_timing' in res:
            timings(dict(request_id=meta['request_id'],batch_id=res['batch_id'],
                actual_batch_size=res['actual_batch_size'],**res['server_timing']))
        return res
    clients=[MeasuredClient(run_id=run_id,robot_id=f'r{i}',reset_call=reset,predict_call=predict,
        sink=emit,clock_domain=domain,deadline_ms=case['deadline_ms'],max_batch_size=case['batch_size'])
        for i in range(case['robots'])]
    try:
        for c in clients:await blocking(c.reset,'episode0')
        server_before=await asyncio.wait_for(handle.stats.remote(),timeout)
        if server_before['execution_kind']!=case['execution_kind']:
            raise ValueError('Execution provenance mismatch')
    except BaseException:
        trace.close();timings.close();pool.shutdown(wait=True)
        raise
    monitor=[]
    inflight=0

    async def phase(warmup):
        spec=case['protocol'];prefix='warmup' if warmup else 'measure'
        minimum=spec[prefix+'_min_s'];maximum=spec[prefix+'_max_s'];required=spec[prefix+'_min_success']
        if not 0<=minimum<=maximum or maximum<=0 or required<1:raise ValueError('Invalid bounded phase protocol')
        start=time.monotonic_ns();stop=asyncio.Event();end=None
        async def one(client,index,scheduled_ns=None):
            nonlocal inflight
            obs=observations[index%len(observations)]
            inflight+=1
            try:
                q=await blocking(lambda:client.predict(obs,warmup=warmup,scheduled_ns=scheduled_ns))
                if q is not None:
                    def execute(_):
                        if case['mode']=='closed_loop':
                            time.sleep(case['action_time_s']);return 1
                        return 0
                    await blocking(client.apply,q,execute)
            except Exception as exc:
                # Requests are already finalized by MeasuredClient. Never
                # suppress them from the denominator or retry the same ID.
                fatal.append(dict(type=type(exc).__name__,message=str(exc)))
                stop.set()
            finally:
                inflight-=1
        async def sleep_until_or_stop(delay):
            if delay<=0:return
            try:await asyncio.wait_for(stop.wait(),timeout=delay)
            except TimeoutError:pass
        async def closed_robot(i,client):
            rng=np.random.default_rng(case['seed']+i)
            cap=case['rate_cap_hz']
            await asyncio.sleep(float(rng.uniform(0,min(case['action_time_s'],1/cap if cap else .01))))
            index=i;next_send=time.monotonic()
            while not stop.is_set():
                if cap:await sleep_until_or_stop(max(0,next_send-time.monotonic()))
                if stop.is_set():break
                sent=time.monotonic();await one(client,index);index+=len(clients)
                if cap:next_send=sent+1/cap
        async def open_loop():
            available=asyncio.Queue()
            for c in clients:available.put_nowait(c)
            tasks=set();seq=0;rate=case['offered_hz']
            if rate is None or not math.isfinite(rate) or not 0<rate<=1e6:raise ValueError('Invalid offered rate')
            async def dispatch(client,index,scheduled):
                try:await one(client,index,scheduled)
                finally:available.put_nowait(client)
            while not stop.is_set():
                scheduled=start+round(seq*1e9/rate)
                await sleep_until_or_stop(max(0,(scheduled-time.monotonic_ns())/1e9))
                if stop.is_set():break
                if available.empty():
                    now=time.monotonic_ns()
                    emit(dict(schema_version='1.0',record_type='request',run_id=run_id,robot_id='overflow',
                        episode_id='episode0',request_id=uuid.uuid4().hex,attempt_id=uuid.uuid4().hex,
                        component_id='s1',observation_seq=seq,clock_domain=domain,scheduled_ns=scheduled,
                        sent_ns=now,finished_ns=now,deadline_ms=case['deadline_ms'],status='rejected',
                        valid_output=False,warmup=warmup,batch_id=None,actual_batch_size=None))
                else:
                    c=available.get_nowait()
                    task=asyncio.create_task(dispatch(c,seq,scheduled));tasks.add(task);task.add_done_callback(tasks.discard)
                seq+=1
            if tasks:await asyncio.gather(*tasks)
        workers=([asyncio.create_task(open_loop())] if case['mode']=='open_loop' else
                 [asyncio.create_task(closed_robot(i,c)) for i,c in enumerate(clients)])
        try:
            while True:
                now=time.monotonic_ns();elapsed=(now-start)/1e9
                success=counts[(warmup,'success')]
                monitor.append(dict(warmup=warmup,elapsed_s=elapsed,successful=success,
                                    finalized=counts[(warmup,'all')],inflight_requests=inflight))
                worker_error=next((t.exception() for t in workers if t.done() and not t.cancelled() and t.exception()),None)
                if worker_error:raise worker_error
                if fatal or elapsed>=maximum or (elapsed>=minimum and success>=required):
                    end=now;break
                await asyncio.sleep(.02)
        finally:
            stop.set()
            await asyncio.gather(*workers)
        return dict(start_ns=start,end_ns=end,duration_s=(end-start)/1e9,
                    success_before_cutoff=success,requirements_met=not fatal and elapsed>=minimum and success>=required,
                    drained_ns=time.monotonic_ns())
    result={'status':'failed','execution_kind':case['execution_kind'],'case_id':case['case_id'],
            'action_semantics':'virtual_timed_chunks' if case['mode']=='closed_loop' else 'no_actions_profile_only',
            'server_before':server_before,'formal_protocol':case['phase']=='formal'}
    try:
        warm=await phase(True);result['warmup']=warm
        if not warm['requirements_met']:
            result['status']='warmup_failed'
        else:
            measured=await phase(False);result['measurement']=measured
            result['status']='complete' if measured['requirements_met'] else 'insufficient_or_failed'
            result['metrics']=evaluate_p1(records,run_id=run_id,clock_domain=domain,
                window_start_ns=measured['start_ns'],window_end_ns=measured['end_ns'],deadline_ms=case['deadline_ms'])
            requests=[r for r in records if r['record_type']=='request' and not r['warmup']
                      and measured['start_ns']<=r['sent_ns']<measured['end_ns']]
            valid=[r for r in requests if r['status']=='ok' and r['valid_output']]
            latency=[(r['finished_ns']-r['sent_ns'])/1e6 for r in valid]
            result['latency_mean_ms']=sum(latency)/len(latency) if latency else None
            result['batch_size_histogram_requests']=dict(Counter(r['actual_batch_size'] for r in valid))
            result['offered_rate_observed_hz']=len(requests)/measured['duration_s']
            result['successful_completion_rate_hz']=len(valid)/measured['duration_s']
            result['capacity_certified']=False
            result['profile_export_allowed']=False
            result['profile_export_note']='Requires independent capacity/queue stability and GPU runtime evidence review'
        result['server_after']=await asyncio.wait_for(handle.stats.remote(),timeout)
        result['errors']=fatal
    finally:
        trace.close();timings.close();pool.shutdown(wait=True)
        write_json(output/'phase_progress.json',monitor)
        write_json(output/'result.json',result)
    return result
