"""Live stateful simulations with stateless shared models and fixed wall windows.
Periodic clocks remain global across episode resets; old episode tasks drain first.
"""
import asyncio,json,math,time,uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from .adapter import JsonlSink,ProtocolError,check_observation_batch_one
from .component_runtime import validate_output
from .execution_plan import runtime_source_manifest,check_runtime_inventory
from .closed_loop_protocol import validate_suite,sample_check,warmup_check
from .paired import load_state_bank,digest_value
from .profile_adoption import inside,read
from .profile_runner import seal
from .regression import write_json,save_observation,load_observation,sha256
from .multicomponent import evaluate_multicomponent


def task_summary(outcomes,start,end):
    result={}
    for cid in dict.fromkeys(x['class_id'] for x in outcomes):
        rows=[x for x in outcomes if x['class_id']==cid];cohort=[x for x in rows if start<=x['start_ns']<end]
        success=sum(x['termination']=='success' and x['terminal_ns'] is not None and x['terminal_ns']<=end for x in cohort)
        result[cid]=dict(all_started=len(rows),warmup_started=sum(x['start_ns']<start for x in rows),measurement_started=len(cohort),
            successful_by_cutoff=success,success_rate_all_measurement_started=success/len(cohort) if cohort else None,
            censored_at_cutoff=sum(x['terminal_ns'] is None or x['terminal_ns']>end for x in cohort),
            termination_counts=dict(Counter(x['termination'] for x in cohort)),
            distinct_state_groups=len({x['group_id'] for x in cohort}),
            repeated_state_exposures=len(cohort)-len({x['group_id'] for x in cohort}),
            actual_steps=sum(x['actual_steps'] for x in cohort),unknown_step_outcomes=sum(x['steps_unknown'] for x in cohort))
    return dict(classes=result,independent_episode_confidence_claim=False,
                denominator='all episodes started inside fixed measurement window; no replacement of censored/failed episodes')


def request_context(role,outcome,recovery_feedback=None):
    context={'execution_history':json.dumps(dict(actual_steps=outcome['actual_steps'],action_chunks=outcome['action_chunks']))}
    if role=='s2' and recovery_feedback:context['task_feedback']=recovery_feedback
    return context


class WarmupShortfall(RuntimeError):pass


async def run_closed_loop(suite,root,transport,simulators,output,*,rep=0):
    validate_suite(suite,root);plan=suite['plan'];p=suite['protocol']
    if type(rep) is not int or not 0<=rep<suite['repetitions']:raise ValueError('Invalid repetition')
    robots={r['robot_id']:(c,r,i) for c in plan['classes'] for i,r in enumerate(c['robots'])}
    if set(simulators)!=set(robots):raise ValueError('Simulator must cover exactly all robots')
    banks={cid:load_state_bank(inside(root,e['directory'])) for cid,e in suite['banks'].items()}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('suite.json',suite),('execution_plan.json',plan),('source_manifest.json',runtime_source_manifest()),
        ('condition.json',dict(suite_id=suite['suite_id'],rep=rep)),('protocol.json',dict(p,source_provenance={cid:dict(manifest_sha256=e['manifest_sha256'],states_sha256=e['states_sha256']) for cid,e in suite['banks'].items()}))]:write_json(out/name,value)
    events=JsonlSink(out/'events.jsonl');requests=JsonlSink(out/'requests.jsonl');details=JsonlSink(out/'request_details.jsonl');life=JsonlSink(out/'lifecycle.jsonl');episode_sink=JsonlSink(out/'episodes.jsonl')
    if plan.get('purpose')=='dedicated_GPU_execution':
        if not hasattr(transport,'launch_gate'):raise ValueError('Dedicated transport launch gate required')
        write_json(out/'dedicated_launch.json',transport.launch_gate)
    records=[];episodes=[];outcomes=[];pending=set();tasks=[];workers={w['worker_id']:w for w in plan['workers']}
    telemetry=JsonlSink(out/'telemetry.jsonl');telemetry_stop=asyncio.Event();sampler=None;telemetry_errors=[]
    run_id='live-'+uuid.uuid4().hex;domain=run_id+'-driver-monotonic';count=0;captured=0;epoch=start=end=None;fatal=None;gate=None;guard=None
    result=dict(status='failed',run_id=run_id,plan_id=plan['plan_id'],execution_mode=plan['mode'],
        action_semantics=p['action_semantics'],formal_performance_claim=False,gpu_kernel_execution_verified=False,
        trial_kind='closed_loop',drained=False,measurement_started=False)
    def emit(row):
        records.append(row);events(row)
        if row['record_type']=='request':requests(row)
    def capture_size(value):
        nonlocal captured
        n=sum(v.nbytes if isinstance(v,np.ndarray) else len(str(v).encode()) for v in value.values())
        if captured+n>p['max_capture_bytes']:raise RuntimeError('Observation/action capture byte budget exceeded')
        captured+=n
    async def inventory():return await asyncio.wait_for(transport.inventory(),p['request_timeout_s'])
    async def sample_telemetry():
        while not telemetry_stop.is_set():
            for wid in workers:
                try:
                    sample=await asyncio.wait_for(transport.telemetry(wid),5) if hasattr(transport,'telemetry') else {'kind':'cpu_inprocess_fixture'}
                    telemetry(dict(worker_id=wid,driver_ns=time.monotonic_ns(),sample=sample))
                    if plan['mode']=='gpu' and (sample.get('kind')!='nvidia_smi_sample' or sample.get('uuid')!=before[wid]['gpu_properties']['uuid']):
                        telemetry_errors.append(wid)
                except Exception as exc:
                    telemetry_errors.append(wid);telemetry(dict(worker_id=wid,error_type=type(exc).__name__))
            try:await asyncio.wait_for(telemetry_stop.wait(),1)
            except TimeoutError:pass
    async def until(t,stop=None):
        delay=max(0,(t-time.monotonic_ns())/1e9)
        if stop:
            try:await asyncio.wait_for(stop.wait(),delay)
            except TimeoutError:pass
        else:await asyncio.sleep(delay)
    async def drain():
        if pending:
            done,remaining=await asyncio.wait(pending,timeout=p['drain_timeout_s'])
            if remaining:raise TimeoutError('Late model work did not drain; do not reuse sessions')
            for task in done:task.result()
    def admit_phase(now):
        nonlocal gate
        if now<start:return
        if gate is None:
            gate=dict(warmup_check(p,plan['classes'],records,start),decision_ns=now)
            write_json(out/'warmup_gate.json',gate);life(dict(kind='warmup_gate',**gate))
            result['measurement_started']=gate['requirements_met']
        if not gate['requirements_met']:raise WarmupShortfall('Warmup success counts insufficient: '+', '.join(gate['shortfall']))
    async def guard_warmup():
        await until(start);admit_phase(time.monotonic_ns())
    try:
        before=await inventory();check_runtime_inventory(plan,before);write_json(out/'workers_before.json',before)
        environments={wid:await asyncio.wait_for(transport.environment(wid),p['request_timeout_s']) for wid in workers} if hasattr(transport,'environment') else {'kind':'cpu_inprocess_fixture'}
        write_json(out/'worker_environments.json',environments)
        simulator_evidence={}
        for rid,sim in simulators.items():
            evidence=await sim.start()
            if evidence['source_manifest']!=runtime_source_manifest() or evidence['kind']!=('libero' if plan['mode']=='gpu' else 'cpu_dummy'):
                raise ValueError('Simulator source/execution provenance differs')
            simulator_evidence[rid]=evidence
        write_json(out/'simulators.json',simulator_evidence)
        sampler=asyncio.create_task(sample_telemetry())
        epoch=time.monotonic_ns();start=epoch+round(p['warmup_s']*1e9);end=start+round(p['duration_s']*1e9)
        life(dict(kind='run_started',run_id=run_id,epoch_ns=epoch,window_start_ns=start,window_end_ns=end))
        async def robot_loop(rid,c,robot,robot_index):
            nonlocal count
            cid=c['audit']['id'];bank,arrays=banks[cid];sim=simulators[rid];specs={x['role']:x for x in c['audit']['components']}
            interval=0 if c['frequency_hz'] is None else math.ceil(1e9/c['frequency_hz']) if c['frequency_hz']>0 else None;next_send=epoch
            for episode_index in range(p['max_episodes_per_robot']):
                if time.monotonic_ns()>=end:return
                admit_phase(time.monotonic_ns())
                case=bank['cases'][(robot_index+rep+episode_index)%len(bank['cases'])];eid=f'e{episode_index:06d}'
                episode=dict(class_id=cid,robot_id=rid,episode_id=eid,start_ns=time.monotonic_ns());episodes.append(episode)
                outcome=dict(episode,case_id=case['case_id'],seed=case['seed'],group_id=digest_value((bank['task'],case['state_sha256'])),
                    state_sha256=case['state_sha256'],actual_steps=0,action_chunks=0,termination='censored',terminal_ns=None,steps_unknown=False)
                state=dict(plan=None,uses=0,safety=None,safety_request_id=None,safety_expiry_ns=0,observation=None,observation_seq=0,
                    safety_observation_version=-1,minimum_safety_observation_version=0,request_versions={},
                    recovery_feedback=None,pending_recovery=None,resume_streak=0)
                stop=asyncio.Event();jobs=[];sessions={};episode_folder=out/'observations'/str(list(robots).index(rid))/eid
                def snapshot(obs):
                    check_observation_batch_one(obs);capture_size(obs)
                    save_observation(episode_folder/f"observation-{state['observation_seq']:05d}",obs,dict(task=bank['task'],case=case,
                        execution_kind='libero_gpu' if plan['mode']=='gpu' else 'cpu_dummy',robot_id=rid,episode_id=eid,
                        actual_steps=outcome['actual_steps'],suite_id=suite['suite_id'],source=bank['source']))
                    state['observation']=deepcopy(obs);state['observation_seq']+=1
                async def request(comp,*,scheduled=None,slot=None):
                    nonlocal count
                    if count>=p['max_requests']:raise RuntimeError('Request cap reached before fixed cutoff')
                    mid=comp['id'];wid=robot['routes'][mid];session=sessions[mid]
                    current_plan=state['plan'];plan_ref=current_plan[0] if current_plan and comp['role']=='s1' else None
                    payload=dict(observation=deepcopy(state['observation']),plan=current_plan[1] if current_plan else None,plan_request_id=plan_ref,
                        context=request_context(comp['role'],outcome,state['recovery_feedback']))
                    meta=dict(plan_id=plan['plan_id'],worker_id=wid,run_id=run_id,robot_id=rid,episode_id=eid,request_id=uuid.uuid4().hex,
                        attempt_id=uuid.uuid4().hex,component_id=mid,observation_seq=session['seq'],generation=session['ack']['generation'])
                    sent=time.monotonic_ns();admit_phase(sent)
                    session['seq']+=1;count+=1;observation_version=state['observation_seq']-1
                    state['request_versions'][meta['request_id']]=observation_version
                    row=dict(schema_version='2.0',record_type='request',**{k:meta[k] for k in ('run_id','robot_id','episode_id','request_id','attempt_id','component_id','observation_seq')},
                        clock_domain=domain,scheduled_ns=sent if scheduled is None else scheduled,sent_ns=sent,finished_ns=sent,
                        deadline_ms=comp['deadline_ms'],status='error',valid_output=False,warmup=sent<start,batch_id=None,actual_batch_size=None,
                        slot_index=slot,plan_request_id=plan_ref)
                    life(dict(kind='request_started',metadata=meta,observation_version=observation_version,input_sha256=digest_value(payload)))
                    response=value=None;error=None
                    task=asyncio.create_task(transport.predict(wid,meta,payload));pending.add(task)
                    def completed(t):
                        pending.discard(t)
                        if not t.cancelled():t.exception()
                    task.add_done_callback(completed)
                    try:
                        response=await asyncio.wait_for(asyncio.shield(task),p['request_timeout_s'])
                        if response['metadata']!=meta or response['instance_id']!=before[wid]['instance_id'] or response['worker_id']!=wid or response['plan_id']!=plan['plan_id']:
                            raise ProtocolError('Model response identity mismatch')
                        if response['status'] not in ('ok','error','timeout','rejected','cancelled'):raise ProtocolError('Unknown model status')
                        row.update(status=response['status'],batch_id=response.get('batch_id'),actual_batch_size=response.get('actual_batch_size'))
                        size=row['actual_batch_size']
                        if (row['batch_id'] is None)!=(size is None) or (size is not None and (type(size) is not int or not 1<=size<=workers[wid]['batch_size'])):
                            raise ProtocolError('Invalid batch evidence')
                        if row['status']=='ok':
                            if size is None:raise ProtocolError('Missing successful batch evidence')
                            validate_output(comp['role'],response['output']);value=response['output'];row['valid_output']=True
                        if row['status'] in ('timeout','rejected','cancelled'):raise TimeoutError('Unresolved model session; stop trial')
                    except TimeoutError:row.update(status='timeout',valid_output=False);error='TimeoutError';raise
                    except asyncio.CancelledError:row.update(status='cancelled',valid_output=False);error='CancelledError';raise
                    except Exception as exc:row.update(status='error',valid_output=False);error=type(exc).__name__;raise
                    finally:
                        row['finished_ns']=time.monotonic_ns();emit(row)
                        details(dict(request_id=meta['request_id'],worker_id=wid,episode_id=eid,observation_version=observation_version,
                            observation_directory=str((episode_folder/f'observation-{observation_version:05d}').relative_to(out)),
                            plan=payload['plan'],plan_request_id=payload['plan_request_id'],context=payload['context'],
                            input_sha256=digest_value(payload),output_sha256=digest_value(value),semantic_output=value if comp['role']!='s1' else None,
                            backend_evidence=response.get('backend_evidence') if response else None,
                            server_timing=response.get('server_timing') if response else None,
                            scheduling_evidence=response.get('scheduling_evidence') if response else None,
                            error_type=error or (response.get('error_type') if response else None)))
                    return row,value
                async def periodic(comp):
                    period=comp['period_ns'];base=epoch+comp['phase_ns'];index=max(0,math.ceil((episode['start_ns']-base)/period))
                    while True:
                        scheduled=base+index*period
                        if scheduled>=end:return
                        await until(scheduled,stop)
                        now=time.monotonic_ns()
                        if now>=end or (stop.is_set() and now<scheduled):return
                        # Complete an already-due obligation with the old
                        # episode's final observation before resetting sessions.
                        current=max(index,(now-base)//period)
                        if current>index:life(dict(kind='periodic_slots_skipped',robot_id=rid,episode_id=eid,component_id=comp['id'],first_slot=index,last_slot=current-1))
                        index=current;scheduled=base+index*period
                        row,value=await request(comp,scheduled=scheduled,slot=index)
                        if comp['role']=='safety':
                            timely=row['finished_ns']<=scheduled+round(comp['deadline_ms']*1e6)
                            state['safety']=value['stop'] if value is not None and timely else None
                            state['safety_request_id']=row['request_id']
                            state['safety_observation_version']=state['request_versions'][row['request_id']]
                            state['safety_expiry_ns']=scheduled+period+round(comp['deadline_ms']*1e6)
                        index+=1
                        if stop.is_set():return
                async def goal():
                    nonlocal next_send
                    if interval is None:await until(end);return
                    while time.monotonic_ns()<end:
                        await until(min(next_send,end))
                        if time.monotonic_ns()>=end:return
                        if 's2' in specs and (state['plan'] is None or state['uses']>=c['audit']['s2_horizon']):
                            row,value=await request(specs['s2'])
                            if value is None:state['plan']=None;next_send=row['sent_ns']+interval;continue
                            state['plan']=(row['request_id'],value['plan']);state['uses']=0
                        if time.monotonic_ns()>=end:return
                        row,action=await request(specs['s1']);state['uses']+=1;next_send=row['sent_ns']+interval
                        if action is None or time.monotonic_ns()>=end:continue
                        has_guard=bool(plan.get('collision_guard')) and 'safety' in specs
                        def visual_blocked():
                            return 'safety' in specs and (state['safety'] is not False or time.monotonic_ns()>=state['safety_expiry_ns'] or
                                state['safety_observation_version']<state['minimum_safety_observation_version'])
                        blocked=visual_blocked();preview=None;limit=min(p['execute_steps'],p['max_steps']-outcome['actual_steps'])
                        if not blocked:
                            if limit>min(v.shape[1] for v in action.values()):raise ProtocolError('Action horizon shorter than declared execution stride')
                            capture_size(action);action_path=out/(row['request_id']+'-action.npz');np.savez_compressed(action_path,**action)
                            life(dict(kind='action_tensor',request_id=row['request_id'],file=action_path.name,sha256=sha256(action_path),requested_steps=limit))
                            if has_guard:
                                preview_start=time.monotonic_ns()
                                preview=await sim.guard(deepcopy(action),limit)
                                life(dict(kind='collision_guard_preview',request_id=row['request_id'],robot_id=rid,episode_id=eid,
                                    start_ns=preview_start,end_ns=time.monotonic_ns(),evidence=preview))
                                from .hybrid_safety import valid_preview
                                blocked=not valid_preview(preview,plan['collision_guard'],digest_value(action),limit)
                                if plan.get('recovery_policy'):
                                    from .recovery_policy import valid_authorization
                                    blocked=not valid_authorization(preview,plan['collision_guard'],plan['recovery_policy'],action,limit,'s1')
                                if time.monotonic_ns()>=end:
                                    life(dict(kind='action_censored_after_preview',request_id=row['request_id']))
                                    return
                        # A new periodic visual result can arrive during the preview.
                        blocked=blocked or visual_blocked()
                        begin=time.monotonic_ns();admit_phase(begin);steps=0;step=None
                        if has_guard:
                            cause=('visual' if visual_blocked() else 'geometry_'+preview['status'] if blocked else 'clear')
                            life(dict(kind='hybrid_safety_gate',request_id=row['request_id'],robot_id=rid,episode_id=eid,
                                checked_at_ns=begin,visual_request_id=state['safety_request_id'],cause=cause,stopped=blocked))
                        life(dict(kind='action_started',request_id=row['request_id'],robot_id=rid,episode_id=eid,start_ns=begin,safety_stopped=blocked))
                        if not blocked:
                            outcome['steps_unknown']=True
                            step=await sim.step(deepcopy(action),limit);steps=step['steps']
                            if has_guard:
                                receipt=step.get('collision_guard')
                                from .hybrid_safety import valid_receipt
                                if not valid_receipt(receipt,preview,steps):raise ProtocolError('Applied action lacks matching collision guard proof')
                                life(dict(kind='collision_guard_applied',request_id=row['request_id'],receipt=receipt))
                            if type(steps) is not int or not 1<=steps<=limit or type(step['success']) is not bool or type(step['done']) is not bool:
                                raise ProtocolError('Invalid simulator step result')
                            outcome['steps_unknown']=False;outcome['actual_steps']+=steps;outcome['action_chunks']+=1
                            snapshot(step['observation'])
                            if state['pending_recovery']:
                                state['resume_streak']+=1
                                if state['resume_streak']==plan['recovery_policy']['resumed_s1_chunks_required']:
                                    life(dict(kind='recovery_s1_resumed',recovery_id=state['pending_recovery'],request_id=row['request_id'],
                                        robot_id=rid,episode_id=eid,at_ns=time.monotonic_ns(),s1_chunks=state['resume_streak']))
                                    state['pending_recovery']=None
                            await until(begin+round(c['action_time_s']*1e9))
                        finished=time.monotonic_ns()
                        emit(dict(schema_version='2.0',record_type='action',**{k:row[k] for k in ('run_id','clock_domain','robot_id','episode_id','request_id','component_id')},
                            action_id=uuid.uuid4().hex,start_ns=begin,end_ns=finished,applied=steps>0,executed_steps=steps,safety_stopped=blocked))
                        if blocked and plan.get('recovery_policy') and preview and preview['status']=='blocked' and not visual_blocked() and time.monotonic_ns()<end:
                            from .recovery_runtime import attempt_retreat
                            step=await attempt_retreat(sim=sim,plan=plan,state=state,outcome=outcome,parent=row,template=action,limit=limit,
                                end=end,out=out,life=life,snapshot=snapshot,capture_size=capture_size,visual_blocked=visual_blocked,until=until,action_time_s=c['action_time_s'])
                            finished=time.monotonic_ns()
                        if step and (step['success'] or step['done'] or outcome['actual_steps']>=p['max_steps']):
                            outcome.update(termination='success' if step['success'] else 'environment_done' if step['done'] else 'step_limit',terminal_ns=finished)
                            return
                try:
                    life(dict(kind='episode_started',**episode,case_id=case['case_id'],seed=case['seed']))
                    reset=await sim.reset(arrays[case['case_id']].copy(),case['seed']);outcome['restored_state_sha256']=reset['restored_state_sha256']
                    snapshot(reset['observation']);outcome['initial_observation_sha256']=digest_value(reset['observation'])
                    if plan.get('prefetch_resets') and episode_index+1<p['max_episodes_per_robot']:
                        next_case=bank['cases'][(robot_index+rep+episode_index+1)%len(bank['cases'])]
                        sim.prepare_reset(arrays[next_case['case_id']].copy(),next_case['seed'])
                        life(dict(kind='simulator_reset_prefetched',robot_id=rid,episode_id=eid,next_case_id=next_case['case_id'],
                            state_sha256=next_case['state_sha256'],started_ns=time.monotonic_ns(),executes_robot_actions=False))
                    for comp,wid in robot['routes'].items():
                        ack=await asyncio.wait_for(transport.reset(wid,run_id,rid,eid),p['request_timeout_s'])
                        expected=dict(run_id=run_id,robot_id=rid,episode_id=eid,worker_id=wid,plan_id=plan['plan_id'],instance_id=before[wid]['instance_id'])
                        if any(ack.get(k)!=v for k,v in expected.items()) or type(ack.get('generation')) is not int:raise ProtocolError('Reset identity mismatch')
                        sessions[comp]=dict(ack=ack,seq=0)
                    if reset['success']:outcome.update(termination='success',terminal_ns=time.monotonic_ns())
                    else:
                        goal_job=asyncio.create_task(goal());jobs=[goal_job]+[asyncio.create_task(periodic(s)) for s in specs.values() if s['role'] in ('safety','monitor')]
                        # Propagate periodic errors immediately, even when goal is still running.
                        async def signal_end():
                            try:await goal_job
                            finally:stop.set()
                        jobs.append(asyncio.create_task(signal_end()));await asyncio.gather(*jobs)
                except BaseException as exc:
                    outcome.update(termination='warmup_shortfall' if isinstance(exc,WarmupShortfall) or (gate is not None and not gate['requirements_met']) else 'error',terminal_ns=time.monotonic_ns());raise
                finally:
                    stop.set()
                    for job in jobs:
                        if not job.done() and not job.cancelling():job.cancel()
                    try:
                        if jobs:await asyncio.gather(*jobs,return_exceptions=True)
                    finally:
                        # A simultaneous gate failure and sibling cancellation
                        # must still finalize every already-started episode.
                        episode['end_ns']=time.monotonic_ns()+1;outcome['end_ns']=episode['end_ns'];outcomes.append(outcome);episode_sink(outcome)
                # No in-flight per-episode requests survive normal gather. Timeout aborts the trial.
            if time.monotonic_ns()<end:raise RuntimeError('Episode cap reached before fixed cutoff')
        tasks=[asyncio.create_task(robot_loop(rid,*value)) for rid,value in robots.items()]
        guard=asyncio.create_task(guard_warmup())
        await asyncio.gather(*tasks,guard);await until(end);await drain()
        after=await inventory();check_runtime_inventory(plan,after)
        if any(after[k]['instance_id']!=before[k]['instance_id'] or after[k]['busy_sessions'] or after[k]['inflight_batches'] for k in before):raise ProtocolError('Worker restarted or not drained')
        write_json(out/'workers_after.json',after);result['drained']=True
    except BaseException as exc:
        fatal=exc;result.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        if guard and not guard.done():guard.cancel()
        if guard:await asyncio.gather(guard,return_exceptions=True)
        for task in tasks:
            if not task.done() and not task.cancelling():task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
        try:await drain()
        except Exception as exc:result['drain_error']=type(exc).__name__
        if isinstance(fatal,WarmupShortfall) and not result.get('drain_error'):
            try:
                after=await inventory();check_runtime_inventory(plan,after)
                if any(after[k]['instance_id']!=before[k]['instance_id'] or after[k]['busy_sessions'] or after[k]['inflight_batches'] for k in before):
                    raise ProtocolError('Worker restarted or did not drain after warmup stop')
                write_json(out/'workers_after.json',after);result['drained']=True
            except Exception as exc:result['drain_error']=type(exc).__name__
        for rid,sim in simulators.items():
            try:await sim.close()
            except Exception as exc:result.setdefault('simulator_cleanup_errors',{})[rid]=type(exc).__name__
        telemetry_stop.set()
        if sampler:await sampler
        result['telemetry_errors']=sorted(set(telemetry_errors))
        if start is not None:
            audit=dict(schema_version='2.0',run_id=run_id,clock_domain=domain,run_epoch_ns=epoch,window_start_ns=start,window_end_ns=end,
                audit_end_ns=max(end,time.monotonic_ns()+1,max((e['end_ns'] for e in episodes),default=0)),
                classes=[c['audit'] for c in plan['classes']],episodes=episodes,records=records)
            try:
                metrics=evaluate_multicomponent(audit);write_json(out/'audit.json',audit);write_json(out/'metrics.json',metrics)
                samples=sample_check(p,plan['classes'],records,start,end);summary=task_summary(outcomes,start,end)
                write_json(out/'sample_check.json',samples);write_json(out/'task_summary.json',summary)
                statuses=dict(Counter(r['status'] for r in records if r['record_type']=='request'))
                result.update(request_statuses=statuses,requests=count,
                    qualified_action_chunks=metrics['qualified_action_chunks'] if result['measurement_started'] else None,
                    weighted_Q=metrics['weighted_Q'] if result['measurement_started'] else None,
                    sample_requirements_met=samples['requirements_met'],formal_protocol_met=samples['formal_protocol_met'] and fatal is None and
                    not result.get('simulator_cleanup_errors') and not telemetry_errors and not result.get('drain_error'))
                if fatal is None and not result.get('simulator_cleanup_errors') and not telemetry_errors:
                    result['status']='measurement_shortfall' if not samples['requirements_met'] else 'complete' if set(statuses)<={'ok'} else 'complete_with_request_failures'
                elif isinstance(fatal,WarmupShortfall) and result['drained'] and not result.get('simulator_cleanup_errors') and not result.get('drain_error') and not telemetry_errors:
                    result['status']='warmup_shortfall'
            except Exception as exc:result.update(audit_error=type(exc).__name__+': '+str(exc),status='failed')
        result['warmup_gate']=gate;result['finished_ns']=time.monotonic_ns()
        result['captured_uncompressed_bytes']=captured;result['episodes_started']=len(outcomes)
        for sink in (events,requests,details,life,episode_sink,telemetry):sink.close()
        write_json(out/'result.json',result);seal(out)
    return result


def reaudit_closed(directory):
    from .profile_runner import verify_evidence
    out=Path(directory);verify_evidence(out);s=read(out/'suite.json');a=read(out/'audit.json');result=read(out/'result.json')
    from .execution_plan import digest,validate_execution_plan
    from .closed_loop_protocol import validate_protocol
    base=dict(s);base.pop('suite_id')
    if digest(base)!=s['suite_id']:raise ValueError('Saved suite identity changed')
    plan=validate_execution_plan(s['plan']);p=validate_protocol(plan,s['protocol']);condition=read(out/'condition.json')
    if plan.get('purpose')=='dedicated_GPU_execution':
        gate=read(out/'dedicated_launch.json')
        if gate.get('mode')=='GPU_dedicated' and plan['joint_residency_review_required']:
            from .dedicated_execution import launch_gate
            if gate!=launch_gate(plan,joint_review=gate.get('joint_review')):raise ValueError('Joint residency authorization changed')
            gate={k:gate[k] for k in ('mode','comparison_eligible')}
        allowed=[dict(mode='cpu_dummy',comparison_eligible=False)] if plan['mode']=='cpu_dummy' else [dict(mode='GPU_joint_residency_calibration',comparison_eligible=False),dict(mode='GPU_dedicated',comparison_eligible=True)]
        if gate not in allowed:raise ValueError('Dedicated launch provenance differs')
    expected_protocol=dict(p,source_provenance={cid:dict(manifest_sha256=e['manifest_sha256'],states_sha256=e['states_sha256']) for cid,e in s['banks'].items()})
    if (condition['suite_id']!=s['suite_id'] or condition['rep'] not in range(s['repetitions']) or
        read(out/'execution_plan.json')!=plan or read(out/'source_manifest.json')!=s['source_manifest'] or read(out/'protocol.json')!=expected_protocol or
        a['classes']!=[c['audit'] for c in plan['classes']] or result['plan_id']!=plan['plan_id'] or result['run_id']!=a['run_id'] or
        a['window_start_ns']-a['run_epoch_ns']!=round(p['warmup_s']*1e9) or
        a['window_end_ns']-a['window_start_ns']!=round(p['duration_s']*1e9)):
        raise ValueError('Closed-loop plan/protocol/source/window binding differs')
    records=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    if records!=a['records']:raise ValueError('Closed-loop raw/audit mismatch')
    if [x for x in records if x['record_type']=='request']!=[json.loads(line) for line in (out/'requests.jsonl').read_text().splitlines()]:raise ValueError('Request journal differs')
    metrics=evaluate_multicomponent(a)
    if metrics!=read(out/'metrics.json'):raise ValueError('Closed-loop metrics changed')
    outcomes=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
    if [{k:x[k] for k in ('class_id','robot_id','episode_id','start_ns','end_ns')} for x in outcomes]!=a['episodes']:
        # Concurrent robots finalize episodes in another order.
        if sorted(({k:x[k] for k in ('class_id','robot_id','episode_id','start_ns','end_ns')} for x in outcomes),key=lambda x:(x['robot_id'],x['episode_id']))!=sorted(a['episodes'],key=lambda x:(x['robot_id'],x['episode_id'])):
            raise ValueError('Episode outcomes/audit mismatch')
    if task_summary(outcomes,a['window_start_ns'],a['window_end_ns'])!=read(out/'task_summary.json'):raise ValueError('Task outcome summary changed')
    samples=sample_check(s['protocol'],s['plan']['classes'],records,a['window_start_ns'],a['window_end_ns'])
    if samples!=read(out/'sample_check.json'):raise ValueError('Sample checks changed')
    gate=read(out/'warmup_gate.json') if (out/'warmup_gate.json').exists() else None
    if gate!=result['warmup_gate'] or result['measurement_started']!=(gate is not None and gate['requirements_met']):raise ValueError('Warmup admission/result differs')
    if gate is not None:
        expected=warmup_check(p,plan['classes'],records,a['window_start_ns'])
        if gate!={**expected,'decision_ns':gate['decision_ns']} or gate['decision_ns']<a['window_start_ns']:raise ValueError('Warmup counts/cutoff differ')
        if not gate['requirements_met']:
            if any((x['sent_ns'] if x['record_type']=='request' else x['start_ns'])>=a['window_start_ns'] for x in records):
                raise ValueError('New request/action started after rejected warmup')
            if result['status'] not in ('warmup_shortfall','failed'):raise ValueError('Warmup shortfall relabelled')
    if result['status']=='warmup_shortfall' and (gate is None or gate['requirements_met'] or not result['drained']):raise ValueError('Invalid warmup stop evidence')
    detail_rows=[json.loads(x) for x in (out/'request_details.jsonl').read_text().splitlines()]
    byrequest={x['request_id']:x for x in records if x['record_type']=='request'};detail_map={x['request_id']:x for x in detail_rows};observations={}
    if len(detail_rows)!=len(detail_map) or set(detail_map)!=set(byrequest):raise ValueError('Request detail coverage differs')
    for detail in detail_rows:
        name=detail['observation_directory']
        if name not in observations:observations[name]=load_observation(inside(out,name))
        obs,manifest=observations[name];r=byrequest[detail['request_id']]
        if any(manifest['provenance'][k]!=r[k] for k in ('robot_id','episode_id')) or detail['episode_id']!=r['episode_id']:
            raise ValueError('Request observation episode differs')
        payload=dict(observation=obs,plan=detail['plan'],plan_request_id=detail['plan_request_id'],context=detail['context'])
        if digest_value(payload)!=detail['input_sha256'] or detail['plan_request_id']!=r['plan_request_id']:raise ValueError('Request input differs from saved observation/context')
        if plan.get('recovery_feedback_roles')==['s2'] and r['component_id']!='s2' and 'task_feedback' in detail['context']:
            raise ValueError('Recovery planning feedback leaked into non-planning model')
    intents=[json.loads(x) for x in (out/'lifecycle.jsonl').read_text().splitlines()]
    tensor_rows={x['request_id']:x for x in intents if x['kind']=='action_tensor'}
    applied=[x for x in records if x['record_type']=='action' and x['applied']]
    for action in applied:
        row=tensor_rows[action['request_id']];path=inside(out,row['file'])
        if sha256(path)!=row['sha256'] or action['executed_steps']>row['requested_steps']:raise ValueError('Applied action tensor/step evidence differs')
        with np.load(path,allow_pickle=False) as data:
            validate_output('s1',dict(data))
            if digest_value(dict(data))!=detail_map[action['request_id']]['output_sha256']:raise ValueError('Applied array differs from model output')
    if plan.get('collision_guard'):
        from .hybrid_safety import audit_guard
        audit_guard(plan,records,intents,detail_map)
    retreats=[];recovery=None
    if plan.get('recovery_policy'):
        from .recovery_audit import audit_recovery
        recovery,retreats=audit_recovery(out,plan,records,intents,detail_map,a['window_start_ns'],a['window_end_ns'])
    if result['status'] in ('complete','complete_with_request_failures','measurement_shortfall'):
        for outcome in outcomes:
            actions=[x for x in applied+retreats if (x['robot_id'],x['episode_id'])==(outcome['robot_id'],outcome['episode_id'])]
            if outcome['steps_unknown'] or outcome['actual_steps']!=sum(x['executed_steps'] for x in actions) or outcome['action_chunks']!=len(actions):
                raise ValueError('Completed episode action accounting differs')
        if not result['drained'] or result.get('telemetry_errors') or result.get('simulator_cleanup_errors') or result.get('drain_error'):
            raise ValueError('Failed cleanup/telemetry relabelled complete')
    if result['sample_requirements_met']!=samples['requirements_met'] or any(result[k]!=(metrics[k] if result['measurement_started'] else None) for k in ('weighted_Q','qualified_action_chunks')):
        raise ValueError('Result differs from recomputed metrics')
    if result['status'] in ('complete','complete_with_request_failures') and not samples['requirements_met']:raise ValueError('Shortfall relabelled complete')
    return dict(recovery=recovery,metrics=metrics,samples=samples,tasks=read(out/'task_summary.json'),actions_revalidated=len(applied),measurement_started=result['measurement_started'],warmup_gate=gate)
