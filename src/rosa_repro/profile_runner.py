"""Bounded component probes over the same transport as R07 serving.
Driver monotonic timestamps, explicit missed slots and immutable raw evidence.
No actions are applied, including when probing S1.
"""
import asyncio
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import uuid
from jsonschema.exceptions import ValidationError

from .gpu_measurement import summarize as summarize_gpu
from .adapter import JsonlSink
from .component_runtime import validate_output
from .execution_plan import check_runtime_inventory,digest,runtime_source_manifest
from .paired import digest_value
from .regression import write_json,sha256
from .profile_protocol import phase_protocol
from .profiling_budget import validate_formal_case
from .probe_evidence import EvidenceWriter


def percentile(values,fraction):
    return sorted(values)[max(0,math.ceil(len(values)*fraction)-1)] if values else None


def metrics(records,window,deadline_ms):
    # Scheduled arrivals determine the open-loop cohort, including missed sends.
    rows=[r for r in records if window['start_ns']<=r['scheduled_ns']<window['end_ns']]
    ids=[r['request_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate measurement request')
    for r in rows:
        if not r['scheduled_ns']<=r['sent_ns']<=r['finished_ns']:raise ValueError('Invalid driver time order')
        if r['status'] not in ('ok','error','timeout','rejected','cancelled','not_sent'):raise ValueError('Unknown status')
    ok=[r for r in rows if r['status']=='ok']
    sent=[r for r in rows if r['status']!='not_sent']
    latency=[(r['finished_ns']-r['sent_ns'])/1e6 for r in ok]
    scheduled_latency=[(r['finished_ns']-r['scheduled_ns'])/1e6 for r in ok]
    duration=(window['end_ns']-window['start_ns'])/1e9
    if duration<=0:raise ValueError('Empty measurement window')
    return dict(arrivals=len(rows),sent=len(sent),successful=len(ok),status_counts=dict(Counter(r['status'] for r in rows)),
        duration_s=duration,arrival_rate_hz=len(rows)/duration,successful_response_cohort_rate_hz=len(ok)/duration,
        completed_before_cutoff=sum(r['status']=='ok' and r['finished_ns']<=window['end_ns'] for r in rows),
        completed_after_cutoff=sum(r['finished_ns']>window['end_ns'] for r in sent),
        latency_ms_successful_only=dict(mean=sum(latency)/len(latency) if latency else None,p50=percentile(latency,.5),p99=percentile(latency,.99)),
        scheduled_latency_ms_successful_only=dict(mean=sum(scheduled_latency)/len(scheduled_latency) if scheduled_latency else None,p99=percentile(scheduled_latency,.99)),
        dispatch_lag_max_ms=max(((r['sent_ns']-r['scheduled_ns'])/1e6 for r in rows),default=0),
        deadline_meet_rate_all_arrivals=sum(v<=deadline_ms for v in scheduled_latency)/len(rows) if rows else 0,
        batch_histogram_requests=dict(Counter(str(r.get('actual_batch_size')) for r in ok)),
        gpu_measurement=summarize_gpu(rows),p99_exploratory=len(ok)<1000,actions_executed=0)


def seal(directory):
    directory=Path(directory)
    files={str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='evidence.json'}
    write_json(directory/'evidence.json',dict(files=files,sha256=digest(files)))


def verify_evidence(directory):
    directory=Path(directory);e=json.loads((directory/'evidence.json').read_text())
    if e['sha256']!=digest(e['files']):raise ValueError('Evidence index changed')
    for name,h in e['files'].items():
        p=(directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or not p.is_file() or sha256(p)!=h:raise ValueError('Evidence missing or modified: '+name)
    for name in ('condition.json','requests.jsonl','result.json','source_manifest.json'):
        if name not in e['files']:raise ValueError('Incomplete evidence')
    return e


def reaudit(directory):
    directory=Path(directory);verify_evidence(directory)
    result=json.loads((directory/'result.json').read_text());condition=json.loads((directory/'condition.json').read_text())
    rows=[json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
    if len(rows)!=result['requests_finalized'] or len({r['request_id'] for r in rows})!=len(rows):raise ValueError('Raw request count/identity mismatch')
    if 'measurement' not in result:raise ValueError('No completed measurement phase')
    for name,window in (('warmup',result['warmup']),('measure',result['measurement'])):
        cohort=[r for r in rows if r['phase']==name and window['start_ns']<=r['scheduled_ns']<window['end_ns']]
        before=sum(r['status']=='ok' and r['finished_ns']<=window['end_ns'] for r in cohort)
        if before!=window['successful_before_cutoff']:raise ValueError('Phase completion count mismatch')
        case=condition['case']
        if case['stage']=='profile':
            scheduled=sorted(r['scheduled_ns'] for r in cohort)
            expected=[];index=0
            while True:
                tick=window['start_ns']+round(index*1e9/case['offered_hz'])
                if tick>=window['end_ns']:break
                expected.append(tick);index+=1
                if index>case['protocol']['max_requests']:raise ValueError('Arrival count exceeds protocol limit')
            if scheduled!=expected:raise ValueError('Planned arrivals missing or altered')
    calculated=metrics([r for r in rows if r['phase']=='measure'],result['measurement'],condition['config']['deadline_ms'])
    if calculated!=result['metrics']:raise ValueError('Raw log/summary disagreement')
    return calculated


async def run_probe(plan,transport,samples,case,output,*,corpus_id,split):
    if not samples:raise ValueError('No observations')
    if case['phase']=='formal':validate_formal_case(case,plan['config'])
    c=plan['config'];w=plan['workers'][0];wid=w['worker_id'];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'condition.json',dict(config=c,case=case,corpus_id=corpus_id,split=split,plan=plan))
    write_json(out/'source_manifest.json',runtime_source_manifest())
    sink=JsonlSink(out/'requests.jsonl');telemetry_sink=JsonlSink(out/'telemetry.jsonl')
    run_id='probe-'+uuid.uuid4().hex;records=[];acks={};pending=set();fatal=[]
    writer=EvidenceWriter()
    result=dict(status='failed',mode=c['mode'],run_id=run_id,clock_domain=run_id+'-driver-monotonic',
        formal_protocol=case['phase']=='formal',formal_performance_claim=False,semantic_accuracy_verified=False,
        capacity_certified=False,actions_executed=0,drained=False)
    def submit(operation):
        try:writer.submit(operation)
        except asyncio.QueueFull:fatal.append('EvidenceQueueFull')
    def emit(row,output=None,has_response=False):
        records.append(row)
        queued=time.monotonic_ns()
        def persist():
            begin=time.monotonic_ns()
            if has_response:row['output_sha256']=digest_value(output)
            if c['role']=='s1' and row['status']=='ok':
                import numpy as np
                name='action-'+row['request_id']+'.npz'
                np.savez_compressed(out/name,**output);row['output_file']=name
            row['persistence_timing']=dict(queue_ms=(begin-queued)/1e6,
                preparation_and_save_ms=(time.monotonic_ns()-begin)/1e6)
            sink(row)
        submit(persist)
    async def bounded(awaitable):return await asyncio.wait_for(awaitable,c['request_timeout_s'])
    stop_telemetry=asyncio.Event()
    async def sample_telemetry():
        while not stop_telemetry.is_set():
            try:
                data=await asyncio.wait_for(transport.telemetry(wid),5)
                row=dict(driver_ns=time.monotonic_ns(),sample=data)
            except Exception as exc:row=dict(driver_ns=time.monotonic_ns(),sample={'kind':'unavailable','error_type':type(exc).__name__})
            submit(lambda row=row:telemetry_sink(row))
            try:await asyncio.wait_for(stop_telemetry.wait(),1.)
            except TimeoutError:pass
    sampler=None
    try:
        # These inputs are fixed for a probe. Hash them before either timed phase.
        def prepare_inputs():
            values=[]
            for sample in samples:
                payload=dict(observation=deepcopy(sample['observation']))
                payload.update(deepcopy(c['payload_overlay']))
                values.append((payload,digest_value(payload)))
            return values
        prepared=await asyncio.to_thread(prepare_inputs)
        before=await bounded(transport.inventory());check_runtime_inventory(plan,before)
        result['worker_before']=before[wid]
        result['worker_environment']=await bounded(transport.environment(wid))
        instance=before[wid]['instance_id']
        for rid in w['members']:acks[rid]=await bounded(transport.reset(wid,run_id,rid,'episode'))
        sequence={rid:0 for rid in w['members']}
        sampler=asyncio.create_task(sample_telemetry())
        async def phase(name):
            p=case['protocol'];minimum=p[name+'_min_s'];maximum=p[name+'_max_s'];required=p[name+'_min_success']
            start=time.monotonic_ns();deadline=start+round(maximum*1e9);end=None;phase_rows=[];serial=0
            available=asyncio.Queue()
            for rid in w['members']:available.put_nowait(rid)
            tasks=set()
            def completed_before(now):return sum(r['status']=='ok' and r['finished_ns']<=now for r in phase_rows)
            def done(now):
                return bool(fatal or writer.errors) or now>=deadline or (now-start>=minimum*1e9 and completed_before(now)>=required)
            def record_base(index,scheduled):
                return dict(request_id=f'{name}-{index}',phase=name,scheduled_ns=scheduled,
                    observation_sha256=samples[index%len(samples)]['sha256'],status='not_sent',actual_batch_size=None,batch_id=None)
            async def one(rid,index,scheduled):
                preparation_begin=time.monotonic_ns()
                row=record_base(index,scheduled);seq=sequence[rid];sequence[rid]+=1
                m=dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run_id,robot_id=rid,episode_id='episode',
                    request_id=row['request_id'],attempt_id=row['request_id'],component_id=c['component_id'],
                    observation_seq=seq,generation=acks[rid]['generation'])
                template,input_sha=prepared[index%len(prepared)]
                payload=deepcopy(template)
                row.update(sent_ns=time.monotonic_ns(),robot_id=rid,metadata=m,input_sha256=input_sha)
                row['driver_timing']=dict(payload_copy_ms=(row['sent_ns']-preparation_begin)/1e6)
                if case['stage']=='profile' and row['sent_ns']-scheduled>c['max_dispatch_lag_s']*1e9:
                    row.update(finished_ns=row['sent_ns'],reason='payload_preparation_dispatch_lag')
                    emit(row);phase_rows.append(row);available.put_nowait(rid);return
                task=asyncio.create_task(transport.predict(wid,m,payload));pending.add(task)
                def finished(t):
                    pending.discard(t)
                    if not t.cancelled():t.exception()  # consume late errors; timeout remains a timeout
                task.add_done_callback(finished)
                reusable=False;response=None
                try:
                    response=await asyncio.wait_for(asyncio.shield(task),c['request_timeout_s'])
                    row['finished_ns']=time.monotonic_ns()
                    if response['metadata']!=m or response['instance_id']!=instance or response['worker_id']!=wid or response['plan_id']!=plan['plan_id']:
                        raise ValueError('Response identity changed')
                    status=response['status']
                    if status not in ('ok','error','timeout','rejected','cancelled'):raise ValueError('Bad response status')
                    if status=='ok':validate_output(c['role'],response['output'])
                    row.update(status=status,batch_id=response.get('batch_id'),actual_batch_size=response.get('actual_batch_size'),
                        server_timing=response.get('server_timing'),backend_evidence=response.get('backend_evidence'),
                        scheduling_evidence=response.get('scheduling_evidence'),
                        error_type=response.get('error_type'))
                    if c['role']!='s1':row['output']=response.get('output')
                    reusable=True
                except TimeoutError:row.update(status='timeout',error_type='ClientTimeout')
                except Exception as exc:
                    row.update(status='error',error_type=type(exc).__name__);fatal.append(type(exc).__name__)
                finally:
                    row.setdefault('finished_ns',time.monotonic_ns())
                    emit(row,deepcopy(response.get('output')) if response else None,response is not None);phase_rows.append(row)
                    # A timed-out slot is retired until the phase is fully drained.
                    if reusable:available.put_nowait(rid)
            def launch(rid,index,scheduled):
                t=asyncio.create_task(one(rid,index,scheduled));tasks.add(t);t.add_done_callback(tasks.discard)
            if case['stage']=='profile':
                rate=case['offered_hz']
                while serial<p['max_requests']:
                    scheduled=start+round(serial*1e9/rate)
                    # Do not backfill missed slots with a burst.
                    await asyncio.sleep(min(.02,max(0,(min(scheduled,deadline)-time.monotonic_ns())/1e9)))
                    now=time.monotonic_ns()
                    if done(now):
                        # Exclude the next arrival only if that boundary still
                        # contains the minimum duration and required responses.
                        # A delayed wakeup can leave `scheduled` before either.
                        if now>=deadline or fatal or writer.errors:end=min(now,deadline)
                        else:
                            sample_end=sorted(r['finished_ns'] for r in phase_rows if r['status']=='ok' and r['finished_ns']<=now)[required-1]
                            end=max(min(now,scheduled),start+math.ceil(minimum*1e9),sample_end)
                        while scheduled<end and serial<p['max_requests']:
                            row=record_base(serial,scheduled);row.update(sent_ns=now,finished_ns=now,reason='phase_cutoff_before_dispatch')
                            emit(row);phase_rows.append(row);serial+=1;scheduled=start+round(serial*1e9/rate)
                        break
                    if now<scheduled:continue
                    if available.empty() or now-scheduled>c['max_dispatch_lag_s']*1e9:
                        row=record_base(serial,scheduled);row.update(sent_ns=now,finished_ns=now,
                            reason='no_free_stream' if available.empty() else 'dispatch_lag')
                        emit(row);phase_rows.append(row)
                    else:launch(available.get_nowait(),serial,scheduled)
                    serial+=1
                else:end=time.monotonic_ns()
            else:
                while serial<p['max_requests']:
                    now=time.monotonic_ns()
                    if done(now):end=now;break
                    if available.empty():await asyncio.sleep(.001);continue
                    launch(available.get_nowait(),serial,now);serial+=1
                else:end=time.monotonic_ns()
            end=end or time.monotonic_ns()
            success_before=completed_before(end)
            if tasks:await asyncio.gather(*tasks)
            # Do not overlap warmup and measured requests, including shielded late work.
            drain_until=time.monotonic()+c['drain_timeout_s']
            while True:
                inventory=await bounded(transport.inventory())
                if inventory[wid]['instance_id']!=instance:raise ValueError('Worker restarted during trial')
                if not pending and inventory[wid]['busy_sessions']==0 and inventory[wid]['inflight_batches']==0:break
                if time.monotonic()>=drain_until:raise TimeoutError('Model requests did not drain')
                await asyncio.sleep(.01)
            return dict(start_ns=start,end_ns=end,duration_s=(end-start)/1e9,successful_before_cutoff=success_before,
                requirements_met=not fatal and end-start>=minimum*1e9 and success_before>=required,
                request_limit_hit=serial>=p['max_requests'],drained_ns=time.monotonic_ns())
        result['warmup']=await phase('warmup')
        if not result['warmup']['requirements_met']:result['status']='warmup_shortfall'
        else:
            result['measurement']=await phase('measure')
            result['metrics']=metrics([r for r in records if r['phase']=='measure'],result['measurement'],c['deadline_ms'])
            result['status']='complete' if result['measurement']['requirements_met'] else 'measurement_shortfall'
        after=await bounded(transport.inventory());check_runtime_inventory(plan,after)
        result.update(worker_after=after[wid],drained=True,errors=fatal)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error_message=str(exc))
    finally:
        stop_telemetry.set()
        if sampler:await sampler
        result['evidence_writer']=await writer.close()
        if writer.errors or 'EvidenceQueueFull' in fatal:
            result.update(status='failed',error_type='EvidencePersistenceFailed',errors=list(fatal)+writer.errors)
        sink.close();telemetry_sink.close()
        result['requests_finalized']=len(records)
        write_json(out/'result.json',result);seal(out)
    return result


def profile_candidate(directories):
    """Conservative R07-shaped candidate, never an automatic scheduler import.

    No capacity bound is inferred from maximum observed throughput. No zero-rate
    row or monotonic smoothing is fabricated. Runtime/semantic review stays explicit.
    """
    trials=[];blocked=[];config=None;identity=None;environment=None;points={};memory=[];seen=set();reps=set();device_runtime=None
    for path in map(Path,directories):
        if path.resolve() in seen:raise ValueError('Duplicate trial directory')
        seen.add(path.resolve())
        e=verify_evidence(path);r=json.loads((path/'result.json').read_text());x=json.loads((path/'condition.json').read_text())
        trials.append(dict(directory=str(path),evidence_sha256=sha256(path/'evidence.json'),status=r['status']))
        if r.get('measurement'):reaudit(path)
        c=x['config'];case=x['case']
        if x.get('plan',{}).get('purpose')=='r05_load_curve_calibration':blocked.append('R05_calibration_not_R07_adoption')
        key=(digest(c),x['corpus_id'],x['split'],digest(json.loads((path/'source_manifest.json').read_text())))
        repetition=(case['offered_hz'],case['rep'])
        if repetition in reps:raise ValueError('Duplicate rate/repetition')
        reps.add(repetition)
        if identity is not None and identity!=key:raise ValueError('Cannot mix configurations, corpora or source versions')
        config=c;identity=key
        if c['mode']!='gpu' or r['mode']!='gpu':blocked.append('cpu_results_are_not_GPU_profiles')
        if case['stage']!='profile' or case['phase']!='formal' or x['split']!='profile':blocked.append('formal_profile_split_required')
        try:validate_formal_case(case,c)
        except (ValueError,KeyError,TypeError,ValidationError):blocked.append('formal_protocol_settings_required')
        if r['status']!='complete' or not r.get('drained'):blocked.append('trial_incomplete_or_not_drained');continue
        m=r['metrics']
        if r['warmup']['duration_s']<30 or r['warmup']['successful_before_cutoff']<20:blocked.append('warmup_shortfall')
        if m['successful']!=m['arrivals'] or m['successful']<1000 or m['duration_s']<120:blocked.append('failures_missed_arrivals_or_sample_shortfall')
        if m['dispatch_lag_max_ms']>c['max_dispatch_lag_s']*1000:blocked.append('load_generator_lag')
        if r['warmup'].get('request_limit_hit') or r['measurement'].get('request_limit_hit'):blocked.append('request_limit_hit')
        worker=r['worker_after'];a=worker['backend_attestation'];props=worker.get('gpu_properties')
        measurement=a.get('gpu_measurement',{})
        if measurement.get('enabled') and measurement.get('unavailable_reason') is None:
            evidence=m.get('gpu_measurement',{})
            if not evidence.get('measured_batches') or evidence.get('unavailable_reasons') or evidence.get('records_missing_evidence'):blocked.append('requested_CUDA_measurement_incomplete')
        if not a.get('model_artifacts_verified'):blocked.append('model_artifact_verification_incomplete')
        if not props or props['name']!=c['hardware_kind']:blocked.append('GPU_hardware_evidence_missing')
        else:memory.append(props['memory_mib'])
        telemetry=[json.loads(line)['sample'] for line in (path/'telemetry.jsonl').read_text().splitlines()]
        if not telemetry or any(t['kind']!='nvidia_smi_sample' for t in telemetry):blocked.append('GPU_telemetry_incomplete')
        else:
            for t in telemetry:
                if not props or t['uuid']!=props.get('uuid') or t['name']!=c['hardware_kind']:blocked.append('GPU_telemetry_identity_mismatch')
                observed=(t['name'],t['driver'],t['total_mib'])
                if device_runtime and observed!=device_runtime:raise ValueError('GPU runtime or memory differs')
                device_runtime=observed
        env=digest(r['worker_environment'])
        if environment and environment!=env:raise ValueError('Worker environments differ')
        environment=env
        if case['offered_hz']:
            points.setdefault(case['offered_hz'],[]).append(dict(mean_ms=m['scheduled_latency_ms_successful_only']['mean'],
                p99_ms=m['scheduled_latency_ms_successful_only']['p99'],sample_count=m['successful'],duration_s=m['duration_s']))
    if not trials:raise ValueError('No trials')
    if any(len(v)<3 for v in points.values()):blocked.append('three_repetitions_per_rate_required')
    if not points:blocked.append('no_profile_points')
    profile=None
    if not blocked:
        profile=dict(id='candidate-'+digest(trials)[:16],component_id=config['component_id'],hardware_kind=config['hardware_kind'],
            model_revision=config['model_revision'],runtime_fingerprint=config['runtime_fingerprint'],source='measured',
            batch_size=config['batch_size'],batch_wait_ms=config['batch_wait_ms'],memory_mib=max(memory),evidence_sha256=digest(trials),
            points=[dict(rate_hz=rate,mean_ms=max(p['mean_ms'] for p in ps),p99_ms=max(p['p99_ms'] for p in ps),
                sample_count=sum(p['sample_count'] for p in ps),duration_s=sum(p['duration_s'] for p in ps)) for rate,ps in sorted(points.items())])
    return dict(schema_version='1.0',status='blocked' if blocked else 'candidate_requires_review',profile=profile,
        blocking_reasons=sorted(set(blocked)),trials=trials,automatic_scheduler_import_allowed=False,
        memory_semantics='full GPU reservation, not measured allocation high-water mark',capacity_bound_inferred=False,
        latency_semantics='scheduled arrival to response; max per-repetition mean and p99',
        review_required=['runtime kernel/dtype/compile evidence and workload representativeness',
                         'queue stability, SLO and semantic model checks; held-out evaluation',
                         'profile provenance and GPU background load before R07 integration'])


def pilot_rate_hints(directory):
    path=Path(directory);m=reaudit(path);x=json.loads((path/'condition.json').read_text());r=json.loads((path/'result.json').read_text())
    if x['case']['stage']!='pilot' or r['status']!='complete' or not r['drained'] or m['arrivals']!=m['successful']:
        raise ValueError('Require a successful, error-free drained pilot')
    rate=m['completed_before_cutoff']/m['duration_s']
    if rate<=0:raise ValueError('No completed requests in measurement interval')
    return dict(config_sha256=digest(x['config']),corpus_id=x['corpus_id'],execution_mode=x['config']['mode'],
        pilot_evidence_sha256=sha256(path/'evidence.json'),observed_completion_rate_hz=rate,
        suggested_rates_hz=[rate*i/10 for i in range(1,11)],capacity_certified=False,
        note='10-100% of this pilot only. Compare several stream counts/durations before choosing a load grid; not a proven maximum or a scheduler capacity bound.')


def compare_probes(left,right):
    """Compare saved outputs on common observations; no semantic equivalence claim."""
    groups=[];conditions=[]
    for directory in (left,right):
        p=Path(directory);verify_evidence(p);conditions.append(json.loads((p/'condition.json').read_text()))
        rows=[json.loads(s) for s in (p/'requests.jsonl').read_text().splitlines()];by_input={}
        for row in rows:
            if row['phase']=='measure' and row['status']=='ok':by_input.setdefault(row['input_sha256'],set()).add(row['output_sha256'])
        groups.append(by_input)
    a,b=conditions
    if a['config']['role']!=b['config']['role'] or a['corpus_id']!=b['corpus_id'] or a['split']!=b['split']:
        raise ValueError('Compare matching roles/corpus/splits')
    common=sorted(set(groups[0])&set(groups[1]))
    if not common:raise ValueError('No common successful inputs')
    return dict(left_config_sha256=digest(a['config']),right_config_sha256=digest(b['config']),
        common_inputs=len(common),left_only_inputs=len(set(groups[0])-set(groups[1])),right_only_inputs=len(set(groups[1])-set(groups[0])),
        rows=[dict(input_sha256=k,left_unique_outputs=len(groups[0][k]),right_unique_outputs=len(groups[1][k]),
            exact_output_sets_equal=groups[0][k]==groups[1][k]) for k in common],
        semantic_equivalence_verified=False,note='Exact output hashes only; stochastic GR00T and different engines need the existing controlled-RNG regression or human semantic review.')
