"""Independent closed-loop clients and fixed windows, with no latency qualification."""
import asyncio,json,math,random,time,uuid
from collections import Counter
from pathlib import Path
from .telemetry_recording import Recorder,summarize as summarize_telemetry
from .gpu_measurement import summarize as summarize_gpu
from .adapter import JsonlSink,check_action_batch_one
from .execution_plan import digest,runtime_source_manifest,check_runtime_inventory
from .profile_protocol import load_corpus
from .r05_protocol import validate_suite,plan_for
from .regression import write_json,sha256


def seal(out):
    files={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='evidence.json'}
    write_json(out/'evidence.json',dict(files=files,sha256=digest(files)))


def verify(out):
    e=json.loads((out/'evidence.json').read_text())
    if e['sha256']!=digest(e['files']) or set(e['files'])!={p.name for p in out.iterdir() if p.is_file() and p.name!='evidence.json'}:raise ValueError('R05 evidence coverage differs')
    for name,h in e['files'].items():
        if Path(name).name!=name or (out/name).is_symlink() or sha256(out/name)!=h:raise ValueError('R05 evidence missing or modified')
    return e


def metrics(events,window):
    phase=window['phase'];start=window['start_ns'];end=window['end_ns'];duration=(end-start)/1e9
    requests={e['request_id']:e for e in events if e['kind']=='request_start' and e['phase']==phase}
    replies={e['request_id']:e for e in events if e['kind']=='request_end' and e['phase']==phase}
    actions=[e for e in events if e['kind']=='action_end' and e['phase']==phase]
    success=[r for r in replies.values() if r['status']=='ok']
    lat=[(r['time_ns']-requests[r['request_id']]['time_ns'])/1e6 for r in success]
    percentile=lambda q:sorted(lat)[max(0,math.ceil(len(lat)*q)-1)] if lat else None
    completed=[a for a in actions if a['time_ns']<=end and a['completed']]
    counts=Counter(r['status'] for r in replies.values());counts['unfinalized']=len(requests)-len(replies)
    return dict(requests=len(requests),status_counts=dict(counts),successful_responses=len(success),
        successful_responses_before_cutoff=sum(r['time_ns']<=end for r in success),
        successful_responses_after_cutoff=sum(r['time_ns']>end for r in success),
        successful_response_cohort_rate_hz=len(success)/duration,
        completed_virtual_action_chunks=len(completed),completed_virtual_action_rate_hz=len(completed)/duration,
        cross_boundary_virtual_actions=sum(a['time_ns']>end for a in actions),
        latency_ms_successful_response_cohort=dict(mean=sum(lat)/len(lat) if lat else None,p50=percentile(.5),p95=percentile(.95),p99=percentile(.99)),
        p99_exploratory=len(lat)<1000,actual_batch_histogram_requests=dict(Counter(str(r['actual_batch_size']) for r in success)),
        gpu_measurement=summarize_gpu(replies.values()),duration_s=duration,latency_slo_ms=None,slo_qualification_applied=False)


async def run_trial(s,corpus_root,cell_id,transport,output,*,pilot_runtime=None):
    validate_suite(s,corpus_root)
    if not s['executable']:raise ValueError('R05 suite timing/sample budget unresolved')
    matches=[x for x in s['cells'] if x['id']==cell_id]
    if len(matches)!=1:raise ValueError('Unknown R05 cell')
    cell=matches[0];plan=plan_for(s['config'],cell);w=plan['workers'][0];wid=w['worker_id'];c=s['config'];pro=s['protocol']
    _,samples=load_corpus(s['corpus'],corpus_root,s['split'],c['mode'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('suite.json',s),('plan.json',plan),('source_manifest.json',runtime_source_manifest())]:write_json(out/name,value)
    telemetry=Recorder(out/'telemetry.jsonl',lambda:transport.telemetry(wid))
    sink=JsonlSink(out/'events.jsonl');events=[];pending={};windows=[];acks={};errors=[];run_id='r05-'+uuid.uuid4().hex
    sequence={r:0 for r in w['members']};sent_count=0;before=after=None;drained=False;measurement_started=False
    def emit(kind,*,before_ns=None,**kw):
        now=time.monotonic_ns()
        if before_ns is not None and now>=before_ns:return None
        row=dict(kind=kind,time_ns=now,**kw);sink(row);events.append(row);return row
    # Model response watchdogs may be intentionally very short in a fault test.
    # Environment/inventory/reset RPCs have a separate derived control allowance.
    async def bounded(coro):return await asyncio.wait_for(coro,max(5.,c['watchdog_s']))
    async def drain():
        if not pending:return True
        tasks=list(pending.values());done,not_done=await asyncio.wait(tasks,timeout=c['drain_s'])
        for rid,task in list(pending.items()):
            if task in done:
                try:
                    reply=task.result();emit('late_response',request_id=rid,status=reply.get('status'),metadata=reply.get('metadata'),
                        batch_id=reply.get('batch_id'),actual_batch_size=reply.get('actual_batch_size'),backend_evidence=reply.get('backend_evidence'))
                except Exception as exc:emit('late_error',request_id=rid,error_type=type(exc).__name__)
                del pending[rid]
        for task in not_done:task.cancel()
        if not_done:await asyncio.gather(*not_done,return_exceptions=True)
        return not not_done
    async def phase(name,seconds):
        nonlocal sent_count
        start=time.monotonic_ns();end=start+round(seconds*1e9);stop=asyncio.Event()
        rng=random.Random(cell['trial_seed']);offsets={r:rng.uniform(0,min(.01,cell['action_time_s'])) for r in w['members']}
        window=dict(phase=name,start_ns=start,end_ns=end,start_offsets_s=offsets);windows.append(window)
        async def robot(index,rid):
            nonlocal sent_count
            await asyncio.sleep(offsets[rid]);iteration=0
            while not stop.is_set() and time.monotonic_ns()<end:
                if sent_count>=pro['max_requests']:
                    errors.append(dict(type='RequestBudgetExceeded',phase=name));stop.set();break
                seq=sequence[rid]
                sample=samples[(index+iteration*cell['robots']+cell['trial_seed'])%len(samples)];iteration+=1
                meta=dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run_id,robot_id=rid,episode_id='episode-0',request_id=uuid.uuid4().hex,
                    attempt_id='attempt-1',component_id='s1',observation_seq=seq,generation=acks[rid]['generation'])
                qid=meta['request_id'];started=emit('request_start',before_ns=end,phase=name,robot_id=rid,request_id=qid,metadata=meta,
                    sample_id=sample['sample_id'],input_sha256=sample['sha256'],phase_iteration=iteration-1)
                if started is None:break
                sequence[rid]+=1;sent_count+=1
                task=asyncio.create_task(transport.predict(wid,meta,dict(observation=sample['observation'])));pending[qid]=task
                response=None;status='error';detail=None
                try:
                    response=await asyncio.wait_for(asyncio.shield(task),c['watchdog_s']);pending.pop(qid,None)
                    if any(response.get(k)!=v for k,v in dict(metadata=meta,plan_id=plan['plan_id'],worker_id=wid,instance_id=before[wid]['instance_id']).items()):
                        raise ValueError('Response route/identity mismatch')
                    status=response['status']
                    if status not in ('ok','error','timeout','rejected','cancelled'):raise ValueError('Unknown response status')
                    if status=='ok':
                        check_action_batch_one(response['output'])
                        if type(response['actual_batch_size']) is not int or not 1<=response['actual_batch_size']<=min(cell['batch_size'],cell['robots']) or not response['batch_id']:
                            raise ValueError('Invalid actual S1 batch size')
                except TimeoutError:status='timeout';detail='operational_watchdog_expired'
                except Exception as exc:status='error';detail=type(exc).__name__
                emit('request_end',phase=name,robot_id=rid,request_id=qid,status=status,
                    batch_id=response.get('batch_id') if response else None,actual_batch_size=response.get('actual_batch_size') if response else None,
                    server_timing=response.get('server_timing') if response else None,backend_evidence=response.get('backend_evidence') if response else None,error=detail)
                if status!='ok':errors.append(dict(type=status,phase=name,request_id=qid));stop.set();break
                if stop.is_set() or time.monotonic_ns()>=end:break
                action=emit('action_start',before_ns=end,phase=name,robot_id=rid,request_id=qid,requested_duration_s=cell['action_time_s'])
                if action is None:break
                # Complete a started virtual action, including a boundary-crossing one; never launch another after cutoff.
                await asyncio.sleep(cell['action_time_s'])
                emit('action_end',phase=name,robot_id=rid,request_id=qid,action_start_ns=action['time_ns'],completed=True)
        robots=[asyncio.create_task(robot(i,r)) for i,r in enumerate(w['members'])]
        outcomes=await asyncio.gather(*robots,return_exceptions=True)
        for value in outcomes:
            if isinstance(value,BaseException):errors.append(dict(type=type(value).__name__,phase=name))
        return window
    try:
        before=await bounded(transport.inventory());check_runtime_inventory(plan,before)
        if before[wid]['model_calls'] or before[wid]['busy_sessions'] or before[wid]['inflight_batches']:raise ValueError('R05 needs a fresh dedicated worker for each trial')
        if pilot_runtime is not None and before[wid]['backend_attestation']!=pilot_runtime:raise ValueError('Confirmation runtime differs from pilot')
        if c['mode']=='gpu' and before[wid]['backend_attestation'].get('model_artifacts_verified') is not True:raise ValueError('Unverified model artifacts')
        write_json(out/'worker_environment.json',await bounded(transport.environment(wid)))
        write_json(out/'telemetry_before.json',await telemetry.sample('before'));telemetry.start()
        for rid in w['members']:
            ack=await bounded(transport.reset(wid,run_id,rid,'episode-0'))
            expected=dict(run_id=run_id,robot_id=rid,episode_id='episode-0',worker_id=wid,plan_id=plan['plan_id'],instance_id=before[wid]['instance_id'])
            if any(ack.get(k)!=v for k,v in expected.items()) or type(ack.get('generation')) is not int or ack['generation']<0:raise ValueError('Reset acknowledgement differs')
            acks[rid]=ack
        warm=await phase('warmup',pro['warmup_s']);drained=await drain()
        warm_ok=metrics(events,warm)['successful_responses_before_cutoff']>=pro['warmup_min_success'] and not errors and drained
        gate=dict(requirements_met=warm_ok,successful_before_cutoff=metrics(events,warm)['successful_responses_before_cutoff'],required=pro['warmup_min_success'])
        write_json(out/'warmup_gate.json',gate)
        if warm_ok:
            measurement_started=True;await phase('measurement',pro['duration_s']);drained=await drain()
    except Exception as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)));drained=await drain()
    finally:
        try:
            after=await bounded(transport.inventory());check_runtime_inventory(plan,after)
            if before and any(after[wid][k]!=before[wid][k] for k in ('instance_id','backend_attestation')):raise ValueError('Worker changed during trial')
            drained=drained and after[wid]['busy_sessions']==after[wid]['inflight_batches']==0
        except Exception as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)));drained=False
        write_json(out/'telemetry_after.json',await telemetry.stop())
        sink.close()
    calculated={window['phase']:metrics(events,window) for window in windows}
    enough=measurement_started and calculated['measurement']['completed_virtual_action_chunks']>=pro['measure_min_success']
    status='failed' if errors or not drained else 'warmup_shortfall' if not measurement_started else 'complete' if enough else 'measurement_shortfall'
    result=dict(schema_version='1.0',kind='r05_no_slo_result',run_id=run_id,cell_id=cell_id,suite_id=s['suite_id'],mode=c['mode'],
        telemetry=summarize_telemetry(telemetry.rows),status=status,measurement_started=measurement_started,windows=windows,metrics=calculated if measurement_started else None,
        warmup_metrics=calculated.get('warmup'),errors=errors,drained=drained,latency_slo_ms=None,watchdog_s=c['watchdog_s'],
        action_semantics='virtual_timed_chunks',physical_actions_executed=0,formal_protocol_met=status=='complete' and pro['phase']=='formal',
        paper_hardware_equivalence_verified=False,task_success_claim=False,worker_before=before,worker_after=after)
    write_json(out/'acks.json',acks);write_json(out/'result.json',result);seal(out);reaudit(out);return result


def reaudit(directory):
    out=Path(directory);evidence=verify(out);read=lambda n:json.loads((out/n).read_text())
    s=read('suite.json');base=dict(s);base.pop('suite_id',None)
    if digest(base)!=s['suite_id']:raise ValueError('Suite digest differs')
    plan=read('plan.json');cell=plan['cell'];result=read('result.json');acks=read('acks.json');pro=s['protocol'];wid=plan['workers'][0]['worker_id']
    if cell not in s['cells'] or plan!=plan_for(s['config'],cell) or read('source_manifest.json')!=s['source_manifest']:raise ValueError('R05 condition/source differs')
    if result['suite_id']!=s['suite_id'] or result['cell_id']!=cell['id'] or result['mode']!=s['config']['mode'] or result['latency_slo_ms'] is not None or result['action_semantics']!='virtual_timed_chunks':raise ValueError('Result binding differs')
    telemetry=[json.loads(line) for line in (out/'telemetry.jsonl').read_text().splitlines()]
    if result['telemetry']!=summarize_telemetry(telemetry):raise ValueError('Telemetry summary differs')
    for stage in ('before','after'):
        path=out/('telemetry_'+stage+'.json');samples=[r['sample'] for r in telemetry if r['stage']==stage]
        if samples!=([read(path.name)] if path.exists() else []):raise ValueError('Telemetry endpoint differs')
    rows=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    if any(a['time_ns']>b['time_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('Driver event clock order differs')
    windows={w['phase']:w for w in result['windows']}
    if list(windows) not in ([],['warmup'],['warmup','measurement']) or len(windows)!=len(result['windows']):raise ValueError('Invalid phases')
    for phase,w in windows.items():
        if w['end_ns']-w['start_ns']!=round(pro['warmup_s' if phase=='warmup' else 'duration_s']*1e9):raise ValueError('Fixed measurement window changed')
    starts={};ends={};actions={};action_ends={};last={};busy={};late=set();phase_iterations={}
    sample_rows=[r for r in s['corpus']['rows'] if r['split']==s['split']]
    for row in rows:
        kind=row['kind'];qid=row['request_id']
        if kind in ('late_response','late_error'):
            if qid not in ends or ends[qid]['status']!='timeout' or qid in late:raise ValueError('Unexpected late completion')
            if kind=='late_response' and row['metadata']!=starts[qid]['metadata']:raise ValueError('Late response identity differs')
            late.add(qid);continue
        phase=row['phase'];rid=row['robot_id'];w=windows[phase]
        if rid not in plan['workers'][0]['members']:raise ValueError('Unknown robot')
        if kind=='request_start':
            meta=row['metadata'];expected=dict(plan_id=plan['plan_id'],worker_id=wid,run_id=result['run_id'],robot_id=rid,episode_id='episode-0',request_id=qid,attempt_id='attempt-1',component_id='s1',observation_seq=last.get(rid,-1)+1,generation=acks[rid]['generation'])
            if qid in starts or busy.get(rid) or meta!=expected or not w['start_ns']<=row['time_ns']<w['end_ns']:raise ValueError('Overlapping, stale or out-of-window request')
            iteration=phase_iterations.get((phase,rid),0)
            if row['phase_iteration']!=iteration:raise ValueError('Replay iteration skipped/repeated')
            phase_iterations[(phase,rid)]=iteration+1
            index=plan['workers'][0]['members'].index(rid);sample=sample_rows[(index+row['phase_iteration']*cell['robots']+cell['trial_seed'])%len(sample_rows)]
            if row['sample_id']!=digest(sample) or row['input_sha256']!=sample['sha256']:raise ValueError('Replay sample differs')
            starts[qid]=row;busy[rid]=qid;last[rid]=meta['observation_seq']
        elif kind=='request_end':
            if qid not in starts or qid in ends or busy.get(rid)!=qid or starts[qid]['phase']!=phase:raise ValueError('Unmatched response')
            if row['status'] not in ('ok','error','timeout','rejected','cancelled'):raise ValueError('Unknown request status')
            if row['status']=='ok' and (type(row['actual_batch_size']) is not int or not 1<=row['actual_batch_size']<=min(cell['robots'],cell['batch_size']) or not row['batch_id']):raise ValueError('Invalid batch size')
            ends[qid]=row;busy[rid]=None
        elif kind=='action_start':
            if qid not in ends or ends[qid]['status']!='ok' or qid in actions or busy.get(rid) or starts[qid]['robot_id']!=rid or starts[qid]['phase']!=phase:raise ValueError('Unmatched or duplicate action')
            if not w['start_ns']<=row['time_ns']<w['end_ns'] or row['requested_duration_s']!=cell['action_time_s']:raise ValueError('Invalid action timing')
            actions[qid]=row;busy[rid]=qid
        elif kind=='action_end':
            if qid not in actions or qid in action_ends or busy.get(rid)!=qid or row['action_start_ns']!=actions[qid]['time_ns'] or row['completed'] is not True:raise ValueError('Unmatched action completion')
            if row['time_ns']-actions[qid]['time_ns']<round(cell['action_time_s']*1e9):raise ValueError('Virtual action wait shortened')
            action_ends[qid]=row;busy[rid]=None
        else:raise ValueError('Unknown event kind')
    if len(starts)>pro['max_requests']:raise ValueError('Request budget exceeded')
    calculated={phase:metrics(rows,w) for phase,w in windows.items()}
    measurement='measurement' in windows
    if result['measurement_started']!=measurement or result['warmup_metrics']!=calculated.get('warmup') or result['metrics']!=(calculated if measurement else None):raise ValueError('Metrics differ from raw events')
    gate_path=out/'warmup_gate.json'
    if gate_path.exists():
        gate=read('warmup_gate.json');count=calculated['warmup']['successful_responses_before_cutoff']
        if gate['successful_before_cutoff']!=count or gate['required']!=pro['warmup_min_success'] or (gate['requirements_met'] and count<gate['required']):raise ValueError('Warmup gate differs')
        if gate['requirements_met'] and any(r['status']!='ok' for r in ends.values() if r['phase']=='warmup'):raise ValueError('Warmup failure ignored')
        if measurement!=gate['requirements_met']:raise ValueError('Measurement started after rejected warmup')
    elif measurement:raise ValueError('Warmup gate absent')
    if measurement:
        if windows['measurement']['start_ns']<max([windows['warmup']['end_ns']]+[r['time_ns'] for r in rows if r.get('phase')=='warmup']):raise ValueError('Warmup work overlapped measurement')
    if result['drained']:
        after=result['worker_after'][wid]
        if after['busy_sessions'] or after['inflight_batches'] or len(starts)!=len(ends) or len(actions)!=len(action_ends):raise ValueError('False drain claim')
    if result['status'] not in ('complete','failed','warmup_shortfall','measurement_shortfall'):raise ValueError('Unknown trial status')
    if result['formal_protocol_met']!=(result['status']=='complete' and pro['phase']=='formal') or result['physical_actions_executed']!=0:raise ValueError('Invalid formal/physical claim')
    if result['status']=='complete':
        for inventory in (result['worker_before'],result['worker_after']):check_runtime_inventory(plan,inventory)
        if any(result['worker_before'][wid][k]!=result['worker_after'][wid][k] for k in ('instance_id','backend_attestation')):raise ValueError('Observed worker runtime changed')
        if not measurement or not result['drained'] or result['errors'] or calculated['measurement']['completed_virtual_action_chunks']<pro['measure_min_success'] or any(r['status']!='ok' for r in ends.values()):raise ValueError('Incomplete experiment marked complete')
        batches={}
        for r in ends.values():batches.setdefault(r['batch_id'],[]).append(r)
        if any(any(r['actual_batch_size']!=len(group) for r in group) for group in batches.values()):raise ValueError('Incomplete/duplicated batch attribution')
        after=result['worker_after'][wid]
        if after['model_calls']!=len(batches) or sorted(after['actual_batch_sizes'])!=sorted(len(g) for g in batches.values()):raise ValueError('Server batch counters differ')
    return dict(status=result['status'],cell=cell,mode=result['mode'],suite_id=s['suite_id'],metrics=result['metrics'],
        measurement_started=measurement,drained=result['drained'],evidence_sha256=evidence['sha256'],latency_slo_ms=None,
        backend_attestation=result['worker_before'][wid]['backend_attestation'] if result['worker_before'] else None)
