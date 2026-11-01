"""Placement-preserving admission and preloaded-standby failover.

A blueprint declares the finite robot identities/model bindings which may join.
Unknown classes or unprofiled rates require a new schedule; they are never
optimistically admitted. This is control-plane evidence, not robot safety proof.
"""
import asyncio,json,math,time,uuid
from copy import deepcopy
from pathlib import Path
from .execution_plan import validate_execution_plan,check_runtime_inventory,digest,runtime_source_manifest
from .heterogeneous import Fleet
from .paired import digest_value
from .component_runtime import validate_output
from .adapter import JsonlSink
from .regression import write_json,sha256
from .profile_runner import seal


def prepare(plan,initial_active,standbys,*,physical_gpu_budget,timeout_s=5.):
    validate_execution_plan(plan)
    if plan.get('purpose') not in (None,'global_population_replan'):raise ValueError('R10 requires a ROSA capped blueprint')
    robots={r['robot_id'] for c in plan['classes'] for r in c['robots']};workers={w['worker_id']:w for w in plan['workers']}
    if len(set(initial_active))!=len(initial_active) or set(initial_active)-robots:raise ValueError('Unknown/duplicate initial robot')
    if not isinstance(standbys,dict) or set(standbys)-set(workers) or len(set(standbys.values()))!=len(standbys):raise ValueError('Unique standby per declared primary required')
    if any(not isinstance(v,str) or not v or v in workers for v in standbys.values()):raise ValueError('Invalid standby identity')
    if type(physical_gpu_budget) is not int or not len(workers)+len(standbys)<=physical_gpu_budget<=128:raise ValueError('Physical budget must include every preloaded standby')
    if type(timeout_s) not in (int,float) or not math.isfinite(timeout_s) or not 0<timeout_s<=600:raise ValueError('Bounded control timeout required')
    allworkers=deepcopy(plan['workers'])
    for primary,spare in standbys.items():allworkers.append(dict(deepcopy(workers[primary]),worker_id=spare))
    result=dict(schema_version='1.0',kind='r10_control_spec',blueprint=deepcopy(plan),initial_active=sorted(initial_active),standbys=deepcopy(standbys),
        physical_gpu_budget=physical_gpu_budget,timeout_s=timeout_s,workers=allworkers,source_manifest=runtime_source_manifest(),
        scope='predeclared robot pool; fixed rates; placement-preserving admission; preloaded standby; global replan is an explicit unresolved outcome')
    result['spec_id']=digest(result);return result


def validate(spec):
    expected=prepare(spec['blueprint'],spec['initial_active'],spec['standbys'],physical_gpu_budget=spec['physical_gpu_budget'],timeout_s=spec['timeout_s'])
    if spec!=expected:raise ValueError('Dynamic control spec/source modified')
    return spec


def envelope(spec):return dict(spec['blueprint'],workers=spec['workers'])


def admission(spec,active):
    """Recompute actual per-worker load, every class SLO/Fmin and robot loop."""
    plan=spec['blueprint'];fleet=Fleet.from_dict(plan['inputs']['fleet']);active=set(active)
    robots={r['robot_id']:(c,r) for c in plan['classes'] for r in c['robots']}
    if active-set(robots):return dict(status='requires_global_replan',reasons=['robot_not_in_blueprint'])
    loads={w['worker_id']:0. for w in plan['workers']};limits={w['worker_id']:[] for w in plan['workers']};estimates={};reasons=[]
    for rid in active:
        c,r=robots[rid]
        for m in c['audit']['components']:
            rate=1e9/m['period_ns'] if m['role'] in ('safety','monitor') else c['frequency_hz']/(c['audit']['s2_horizon'] if m['role']=='s2' else 1)
            loads[r['routes'][m['id']]]+=rate;limits[r['routes'][m['id']]].append(m['deadline_ms'])
    for w in plan['workers']:
        wid=w['worker_id']
        if not limits[wid]:continue
        e=fleet.profiles[w['profile_id']].at(loads[wid]);estimates[wid]=e
        if e.status!='feasible' or e.p99_ms>min(limits[wid]):reasons.append(wid+':profile_or_SLO')
    for c in plan['classes']:
        count=sum(r['robot_id'] in active for r in c['robots']);minimum=fleet.classes[c['audit']['id']]['min_action_rate_hz']
        if count*c['frequency_hz']<minimum:reasons.append(c['audit']['id']+':Fmin')
    for rid in active:
        c,r=robots[rid];loop=c['action_time_s']
        for m in c['audit']['components']:
            if m['role'] in ('s1','s2'):
                e=estimates[r['routes'][m['id']]]
                if e.mean_ms is None:continue
                loop+=e.mean_ms/1000/(c['audit']['s2_horizon'] if m['role']=='s2' else 1)
        if c['frequency_hz']*loop>1+1e-12:reasons.append(rid+':closed_loop')
    return dict(status='admitted' if not reasons else 'requires_global_replan',reasons=reasons,primary_loads_hz=loads,
        scope='profile-conditional fixed-rate feasibility, not a measured performance guarantee')


class Controller:
    def __init__(self,spec,transport,output):
        validate(spec);self.spec=spec;self.plan=spec['blueprint'];self.transport=transport;self.timeout=spec['timeout_s']
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=False);write_json(self.out/'spec.json',spec)
        self.sink=JsonlSink(self.out/'events.jsonl');self.events=[];self.lock=asyncio.Lock();self.run_id='control-'+uuid.uuid4().hex
        self.robots={r['robot_id']:(c,r) for c in self.plan['classes'] for r in c['robots']};self.state={};self.active=set();self.routes={rid:deepcopy(r['routes']) for rid,(c,r) in self.robots.items()}
        self.failed=set();self.promoted=set();self.pending=set();self.actions={};self.tickets={};self.inventory={};self.closed=False;self.ready=False
    def emit(self,kind,**kw):
        row=dict(kind=kind,time_ns=time.monotonic_ns(),**kw);self.events.append(row);self.sink(row);return row
    async def start(self):
        self.emit('started',run_id=self.run_id,spec_id=self.spec['spec_id'])
        self.inventory=await asyncio.wait_for(self.transport.inventory(),self.timeout)
        check_runtime_inventory(envelope(self.spec),self.inventory)
        for primary,spare in self.spec['standbys'].items():
            if self.inventory[primary]['backend_attestation']!=self.inventory[spare]['backend_attestation']:raise ValueError('Standby observed model/runtime differs')
        initial=admission(self.spec,self.spec['initial_active'])
        if initial['status']!='admitted':raise ValueError('Initial population is not profile-feasible')
        write_json(self.out/'inventory.json',self.inventory)
        # Admit the initial set atomically for positive Fmin; requests start only after all resets.
        for rid in self.spec['initial_active']:await self._reset(rid,0)
        self.active.update(self.spec['initial_active']);self.ready=True;self.emit('initial_active',robots=sorted(self.active),decision=initial)
    async def _reset(self,rid,epoch):
        eid=uuid.uuid4().hex;acks={};c,_=self.robots[rid]
        for mid,wid in self.routes[rid].items():
            ack=await asyncio.wait_for(self.transport.reset(wid,self.run_id,rid,eid),self.timeout)
            expected=dict(run_id=self.run_id,robot_id=rid,episode_id=eid,worker_id=wid,plan_id=self.plan['plan_id'],instance_id=self.inventory[wid]['instance_id'])
            if any(ack.get(k)!=v for k,v in expected.items()) or type(ack.get('generation')) is not int:raise ValueError('Dynamic reset identity changed')
            acks[mid]=ack
        self.state[rid]=dict(epoch=epoch,episode=eid,acks=acks,seq={m:0 for m in acks},paused=False,unsafe=False,plan=None,uses=0,safety=None,next_send_ns=0,latest_s1=None,s1_pending=None)
        row=self.emit('reset',robot_id=rid,epoch=epoch,episode_id=eid,routes=deepcopy(self.routes[rid]),acks=acks)
        self.state[rid]['start_ns']=row['time_ns']
    async def join(self,rid):
        async with self.lock:
            if rid in self.active:return dict(status='already_active' if not self.state[rid]['paused'] else 'blocked')
            decision=admission(self.spec,self.active|{rid})
            if decision['status']!='admitted':self.emit('admission_rejected',robot_id=rid,decision=decision);return decision
            if any(w in self.failed for w in self.routes[rid].values()):return dict(status='blocked_failed_route')
            if rid in self.actions:return dict(status='blocked_action_inflight')
            if self.state.get(rid,{}).get('unsafe'):return dict(status='blocked_unknown_action')
            await self._reset(rid,self.state.get(rid,{}).get('epoch',-1)+1);self.active.add(rid)
            self.emit('joined',robot_id=rid,decision=decision);return decision
    async def leave(self,rid):
        async with self.lock:
            if rid not in self.active:return dict(status='not_active')
            decision=admission(self.spec,self.active-{rid})
            if decision['status']!='admitted':return decision
            self.active.remove(rid);s=self.state[rid];s['epoch']+=1;s['paused']=True;s['plan']=None
            self.emit('left',robot_id=rid,epoch=s['epoch']);return dict(status='left')
    async def predict(self,rid,mid,observation,context=None,*,scheduled_ns=None,slot_index=None):
        async with self.lock:
            if rid not in self.active or self.state[rid]['paused'] or self.state[rid]['unsafe']:return dict(status='blocked')
            s=self.state[rid];c,_=self.robots[rid];m=next((m for m in c['audit']['components'] if m['id']==mid),None)
            if m is None:raise ValueError('Unknown component')
            if m['role']=='s1':
                if rid in self.actions:return dict(status='blocked_action_inflight')
                if s['s1_pending']:return dict(status='blocked_prediction_inflight')
                if c['frequency_hz']<=0:return dict(status='rate_disabled')
                if time.monotonic_ns()<s['next_send_ns']:return dict(status='rate_limited',retry_after_s=(s['next_send_ns']-time.monotonic_ns())/1e9)
            if m['role']=='s1' and any(x['role']=='s2' for x in c['audit']['components']) and (not s['plan'] or s['uses']>=c['audit']['s2_horizon']):return dict(status='needs_plan')
            if scheduled_ns is not None and (type(scheduled_ns) is not int or not s['start_ns']<=scheduled_ns<=time.monotonic_ns() or m['role'] not in ('safety','monitor') or type(slot_index) is not int or slot_index<0):raise ValueError('Invalid periodic dispatch time')
            if scheduled_ns is None and slot_index is not None:raise ValueError('Periodic slot needs scheduled time')
            wid=self.routes[rid][mid];epoch=s['epoch'];qid=uuid.uuid4().hex
            if m['role']=='s1':
                s['next_send_ns']=time.monotonic_ns()+round(1e9/c['frequency_hz']);s['latest_s1']=qid;s['s1_pending']=qid;s['uses']+=1
            captured_plan=deepcopy(s['plan']) if m['role']=='s1' else None
            meta=dict(plan_id=self.plan['plan_id'],worker_id=wid,run_id=self.run_id,robot_id=rid,episode_id=s['episode'],request_id=qid,attempt_id=qid,
                component_id=mid,observation_seq=s['seq'][mid],generation=s['acks'][mid]['generation']);s['seq'][mid]+=1
            payload=dict(observation=deepcopy(observation),plan=s['plan']['text'] if s['plan'] and m['role']=='s1' else None,context=context or {})
            begin=self.emit('prediction_started',robot_id=rid,epoch=epoch,metadata=meta,input_sha256=digest_value(payload),deadline_ms=m['deadline_ms'],scheduled_ns=scheduled_ns,slot_index=slot_index,plan_request_id=captured_plan['request_id'] if captured_plan else None)
            task=asyncio.create_task(self.transport.predict(wid,meta,payload));self.pending.add(task)
        try:
            response=await asyncio.wait_for(asyncio.shield(task),self.timeout)
            async with self.lock:
                current=self.state[rid];valid=rid in self.active and not current['paused'] and current['epoch']==epoch and self.routes[rid][mid]==wid
                if any(response.get(k)!=v for k,v in dict(metadata=meta,worker_id=wid,plan_id=self.plan['plan_id'],instance_id=self.inventory[wid]['instance_id']).items()):raise ValueError('Dynamic response identity differs')
                status=response['status'] if valid else 'stale';output=response.get('output')
                if status=='ok':validate_output(m['role'],output)
                done=self.emit('prediction_finished',request_id=qid,robot_id=rid,epoch=epoch,status=status,worker_id=wid,
                    output_sha256=digest_value(output) if status=='ok' else None,actual_batch_size=response.get('actual_batch_size'),batch_id=response.get('batch_id'),
                    backend_evidence=response.get('backend_evidence'),semantic_output=output if m['role']!='s1' and status=='ok' else None)
                timely=done['time_ns']-begin['time_ns']<=m['deadline_ms']*1e6
                if status=='ok':
                    self.tickets[qid]=dict(robot_id=rid,epoch=epoch,component=mid,role=m['role'],output=deepcopy(output),used=False,timely=timely,
                        plan_timely=captured_plan['timely'] if captured_plan else True,output_sha256=digest_value(output))
                    if m['role']=='s2':current['plan']=dict(text=output['plan'],request_id=qid,timely=timely);current['uses']=0
                    if m['role']=='safety':
                        due=scheduled_ns if scheduled_ns is not None else begin['time_ns']
                        current['safety']=dict(stop=output['stop'] if done['time_ns']<=due+m['deadline_ms']*1e6 else None,expires_ns=due+m['period_ns']+round(m['deadline_ms']*1e6))
                return dict(status=status,request_id=qid,ticket_id=qid if status=='ok' else None,output=deepcopy(output) if status=='ok' else None)
        except asyncio.CancelledError:
            self.emit('prediction_failed',request_id=qid,robot_id=rid,epoch=epoch,error_type='CancelledError');raise
        except Exception as exc:
            self.emit('prediction_failed',request_id=qid,robot_id=rid,epoch=epoch,error_type=type(exc).__name__);return dict(status='timeout' if isinstance(exc,TimeoutError) else 'error',request_id=qid)
        finally:
            async with self.lock:
                if self.state[rid].get('s1_pending')==qid:self.state[rid]['s1_pending']=None
            if task.done():self.pending.discard(task)
            else:
                def finished(t):
                    self.pending.discard(t)
                    try:t.result();status='discarded_late_response'
                    except BaseException:status='late_error'
                    if not self.closed:self.emit('late_completion',request_id=qid,status=status)
                task.add_done_callback(finished)
    async def apply(self,ticket_id,executor):
        async with self.lock:
            ticket=self.tickets.get(ticket_id)
            if not ticket or ticket['role']!='s1' or ticket['used']:return dict(status='rejected_duplicate_or_unknown')
            rid=ticket['robot_id'];s=self.state[rid];c,_=self.robots[rid];now=time.monotonic_ns()
            if rid not in self.active or s['paused'] or s['unsafe'] or s['epoch']!=ticket['epoch'] or s['latest_s1']!=ticket_id:return dict(status='rejected_stale')
            if rid in self.actions:return dict(status='rejected_busy_or_rate')
            safety=s['safety']
            if any(m['role']=='safety' for m in c['audit']['components']) and (not safety or safety['stop'] is not False or now>=safety['expires_ns']):
                ticket['used']=True;self.emit('action_stopped',ticket_id=ticket_id,robot_id=rid,epoch=ticket['epoch'],reason='safety',output_sha256=ticket['output_sha256'])
                return dict(status='safety_blocked')
            ticket['used']=True
            self.emit('action_started',ticket_id=ticket_id,robot_id=rid,epoch=ticket['epoch'],output_sha256=ticket['output_sha256'])
            task=asyncio.create_task(executor(rid,deepcopy(ticket['output']),c['action_time_s']));self.actions[rid]=task
        try:
            steps=await asyncio.wait_for(asyncio.shield(task),self.timeout+c['action_time_s'])
            if type(steps) is not int or not 0<=steps<=min(v.shape[1] for v in ticket['output'].values()):raise ValueError('Invalid executor step count')
            self.emit('action_finished',ticket_id=ticket_id,robot_id=rid,status='completed',steps=steps,
                s1_s2_deadlines_met=ticket['timely'] and ticket['plan_timely'],pipeline=c['audit']['pipeline'])
            return dict(status='completed',steps=steps)
        except asyncio.CancelledError:
            async with self.lock:self.state[rid]['unsafe']=True;self.state[rid]['paused']=True
            self.emit('action_finished',ticket_id=ticket_id,robot_id=rid,status='unknown',steps=None,error_type='CancelledError');raise
        except Exception as exc:
            async with self.lock:self.state[rid]['unsafe']=True;self.state[rid]['paused']=True
            self.emit('action_finished',ticket_id=ticket_id,robot_id=rid,status='unknown',steps=None,error_type=type(exc).__name__)
            return dict(status='unknown',requires_external_recovery_review=True)
        finally:
            if task.done():self.actions.pop(rid,None)
            else:
                def done(t):
                    self.actions.pop(rid,None)
                    try:t.result()
                    except BaseException:pass
                    if not self.closed:self.emit('late_action_completion',ticket_id=ticket_id,robot_id=rid,status='prior_unknown_not_relabelled')
                task.add_done_callback(done)
    async def failover(self,worker_id):
        async with self.lock:
            affected=sorted(r for r in self.active if worker_id in self.routes[r].values())
            if worker_id in self.failed:return dict(status='already_quarantined')
            if worker_id not in {w['worker_id'] for w in self.spec['workers']}:raise ValueError('Unknown failed worker')
            self.failed.add(worker_id);began=self.emit('worker_quarantined',worker_id=worker_id,robots=affected)
            for rid in affected:
                s=self.state[rid];s['epoch']+=1;s['paused']=True;s['plan']=None;s['safety']=None
            spare=self.spec['standbys'].get(worker_id)
            if not spare or spare in self.failed or spare in self.promoted:
                self.emit('failover_blocked',worker_id=worker_id,reason='no_healthy_preloaded_standby');return dict(status='blocked_no_standby')
        active_actions=[self.actions[r] for r in affected if r in self.actions]
        if active_actions:
            _,pending=await asyncio.wait(active_actions,timeout=self.timeout)
            if pending:
                async with self.lock:
                    for rid in affected:self.state[rid]['unsafe']=True
                    self.emit('failover_blocked',worker_id=worker_id,reason='action_completion_unknown')
                return dict(status='blocked_unknown_action')
        async with self.lock:
            try:
                observed=await asyncio.wait_for(self.transport.worker_stats(spare),self.timeout)
                if any(observed.get(k)!=self.inventory[spare].get(k) for k in ('instance_id','backend_attestation','worker_spec_sha256','runtime_source_sha256')):raise ValueError('Standby instance/runtime changed')
                for rid in affected:
                    if self.state[rid]['unsafe']:raise ValueError('Action state requires recovery review')
                # Update inactive blueprint robots too; future joins use the promoted route.
                for rid in self.routes:
                    self.routes[rid]={m:(spare if wid==worker_id else wid) for m,wid in self.routes[rid].items()}
                for rid in affected:await self._reset(rid,self.state[rid]['epoch'])
                self.promoted.add(spare)
                done=self.emit('failover_ready',worker_id=worker_id,standby_id=spare,robots=affected,latency_ns=time.monotonic_ns()-began['time_ns'])
                return dict(status='ready',affected=affected,standby=spare,latency_ns=done['latency_ns'])
            except Exception as exc:
                for rid in affected:self.state[rid]['paused']=True
                self.emit('failover_blocked',worker_id=worker_id,reason=type(exc).__name__);return dict(status='blocked_standby_validation')
    async def healthcheck(self):
        results={}
        for w in self.spec['workers']:
            wid=w['worker_id']
            if wid in self.failed:continue
            try:
                row=await asyncio.wait_for(self.transport.worker_stats(wid),self.timeout)
                if row['instance_id']!=self.inventory[wid]['instance_id']:raise ValueError('Worker restarted')
                results[wid]='healthy'
            except Exception:results[wid]='failed'
        self.emit('healthcheck',workers=results);return results
    async def close(self,*,reason='closed'):
        tasks=list(self.pending|set(self.actions.values()));unresolved=0
        if tasks:
            _,pending=await asyncio.wait(tasks,timeout=self.timeout);unresolved=len(pending)
            for t in pending:t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        self.emit('closed',close_reason=reason,initialized=self.ready,unresolved_calls=unresolved,active=sorted(self.active),unrecovered_robots=sorted(r for r in self.active if self.state[r]['paused'] or self.state[r]['unsafe']),unsafe_robots=sorted(r for r,s in self.state.items() if s['unsafe']))
        self.closed=True;self.sink.close();summary=metrics(self.events);write_json(self.out/'result.json',summary);seal(self.out);reaudit(self.out);return summary


def metrics(rows):
    starts={};finished={};actions={};ends={};epochs={};retired=set();active=set();failures=[];changes=[];latest={};busy=set()
    if not rows or rows[0]['kind']!='started' or rows[-1]['kind']!='closed':raise ValueError('Control trace incomplete')
    if any(a['time_ns']>b['time_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('Control clock order differs')
    for r in rows:
        kind=r['kind']
        if kind=='reset':epochs[r['robot_id']]=r['epoch']
        elif kind=='initial_active':active=set(r['robots'])
        elif kind=='joined':active.add(r['robot_id'])
        elif kind=='left':active.discard(r['robot_id']);epochs[r['robot_id']]=r['epoch']
        elif kind=='quiesced':
            for rid in r['robots']:epochs[rid]+=1
        elif kind=='worker_quarantined':
            retired.add(r['worker_id']);failures.append(r)
            for rid in r['robots']:epochs[rid]+=1
        elif kind=='failover_ready':changes.append(r)
        elif kind=='prediction_started':
            qid=r['metadata']['request_id']
            if qid in starts or r['robot_id'] not in active or r['epoch']!=epochs[r['robot_id']] or r['metadata']['worker_id'] in retired:raise ValueError('Invalid dynamic dispatch')
            starts[qid]=r
            latest[(r['robot_id'],r['metadata']['component_id'])]=qid
        elif kind in ('prediction_finished','prediction_failed'):
            qid=r['request_id']
            if qid not in starts or qid in finished:raise ValueError('Unmatched/repeated dynamic response')
            if r.get('status')=='ok' and (r['epoch']!=epochs[r['robot_id']] or r['robot_id'] not in active):raise ValueError('Stale dynamic response accepted')
            finished[qid]=r
        elif kind=='action_started':
            qid=r['ticket_id'];response=finished.get(qid)
            if qid in actions or not response or response.get('status')!='ok' or r['epoch']!=epochs[r['robot_id']] or r['robot_id'] not in active or r['output_sha256']!=response['output_sha256']:raise ValueError('Stale/duplicate/unmatched action applied')
            if qid!=latest.get((r['robot_id'],starts[qid]['metadata']['component_id'])) or r['robot_id'] in busy:raise ValueError('Superseded/overlapping action')
            actions[qid]=r;busy.add(r['robot_id'])
        elif kind=='action_finished':
            qid=r['ticket_id']
            if qid not in actions or qid in ends or (r['status']=='unknown' and r['steps'] is not None):raise ValueError('Invalid action completion')
            if r['robot_id']!=actions[qid]['robot_id']:raise ValueError('Action robot differs')
            if r.get('pipeline')=='P1' and r['status']=='completed':
                timely=finished[qid]['time_ns']-starts[qid]['time_ns']<=starts[qid]['deadline_ms']*1e6
                if r['s1_s2_deadlines_met']!=timely:raise ValueError('P1 deadline flag differs')
            ends[qid]=r;busy.remove(r['robot_id'])
    complete=[r for r in ends.values() if r['status']=='completed' and r['steps']>0]
    duration=(rows[-1]['time_ns']-rows[0]['time_ns'])/1e9
    qualified=sum(r.get('pipeline')=='P1' and r.get('s1_s2_deadlines_met') for r in complete)
    segments=[]
    for failure in failures:
        recovery=next((r for r in changes if r['worker_id']==failure['worker_id'] and r['time_ns']>=failure['time_ns']),None)
        segments.append(dict(worker_id=failure['worker_id'],affected_robots=failure['robots'],ready_after_ns=recovery['time_ns']-failure['time_ns'] if recovery else None,
            first_completed_action_after_ready_ns=min((r['time_ns']-recovery['time_ns'] for r in complete if recovery and r['robot_id'] in failure['robots'] and actions[r['ticket_id']]['time_ns']>=recovery['time_ns']),default=None)))
    for segment,failure in zip(segments,failures):
        recovery=next((r for r in changes if r['worker_id']==failure['worker_id'] and r['time_ns']>=failure['time_ns']),None)
        boundaries=[('before_fault',rows[0]['time_ns'],failure['time_ns']),('during_fault',failure['time_ns'],recovery['time_ns'] if recovery else rows[-1]['time_ns'])]
        if recovery:boundaries.append(('after_ready',recovery['time_ns'],rows[-1]['time_ns']))
        segment['windows']=[]
        for name,start,end in boundaries:
            values=[r for r in complete if r['robot_id'] in failure['robots'] and start<=r['time_ns']<end]
            n=sum(r.get('pipeline')=='P1' and r.get('s1_s2_deadlines_met') for r in values);seconds=(end-start)/1e9
            segment['windows'].append(dict(name=name,start_ns=start,end_ns=end,completed_actions=len(values),qualified_P1_actions=n,qualified_P1_rate_hz=n/seconds if seconds else None))
    retired=rows[-1].get('close_reason')=='reconfigured' and rows[-1]['initialized'] and not rows[-1]['unresolved_calls'] and len(starts)==len(finished) and len(actions)==len(ends) and not rows[-1]['unsafe_robots']
    return dict(kind='r10_control_result',status='retired' if retired else 'complete' if rows[-1]['initialized'] and not rows[-1]['unresolved_calls'] and len(starts)==len(finished) and len(actions)==len(ends) and not rows[-1]['unsafe_robots'] and not rows[-1]['unrecovered_robots'] else 'incomplete',
        service_ready=rows[-1]['initialized'] and not rows[-1]['unrecovered_robots'],unrecovered_robots=rows[-1]['unrecovered_robots'],predictions=len(starts),accepted_predictions=sum(r.get('status')=='ok' for r in finished.values()),stale_responses=sum(r.get('status')=='stale' for r in finished.values()),
        actions_started=len(actions),actions_completed=len(complete),unknown_actions=sum(r['status']=='unknown' for r in ends.values()),
        qualified_P1_action_chunks=qualified,qualified_P1_action_rate_hz=qualified/duration if duration else None,duration_s=duration,failovers=segments,
        full_P2_P4_qualified_metric_claim=False,robot_safety_certified=False,physical_gpu_failure_claim=False,
        scope='control-plane trace; P1 rate over the entire control trial; no task-success or steady-state capacity claim')


def reaudit(directory):
    p=Path(directory);e=json.loads((p/'evidence.json').read_text())
    if e['sha256']!=digest(e['files']):raise ValueError('Control evidence index changed')
    if set(e['files'])!={str(x.relative_to(p)) for x in p.rglob('*') if x.is_file() and x.name!='evidence.json'}:raise ValueError('Control evidence inventory differs')
    for name,h in e['files'].items():
        f=(p/name).resolve()
        if not f.is_relative_to(p.resolve()) or sha256(f)!=h:raise ValueError('Control evidence changed')
    if not {'spec.json','events.jsonl','result.json'}<=set(e['files']):raise ValueError('Control evidence incomplete')
    spec=json.loads((p/'spec.json').read_text());validate(spec)
    rows=[json.loads(s) for s in (p/'events.jsonl').read_text().splitlines()];value=metrics(rows)
    if rows[0]['spec_id']!=spec['spec_id']:raise ValueError('Control trace uses other spec')
    robots={r['robot_id']:(c,r) for c in spec['blueprint']['classes'] for r in c['robots']};resets={};requests={}
    if rows[-1]['initialized']:
        inventory=json.loads((p/'inventory.json').read_text());check_runtime_inventory(envelope(spec),inventory)
    for row in rows:
        if row['kind']=='reset':resets[row['robot_id']]=row
        elif row['kind']=='prediction_started':
            rid=row['robot_id'];meta=row['metadata'];c,_=robots[rid];reset=resets[rid]
            m=next(x for x in c['audit']['components'] if x['id']==meta['component_id']);wid=reset['routes'][m['id']]
            expected=dict(plan_id=spec['blueprint']['plan_id'],worker_id=wid,run_id=rows[0]['run_id'],robot_id=rid,episode_id=reset['episode_id'],generation=reset['acks'][m['id']]['generation'])
            if any(meta.get(k)!=v for k,v in expected.items()) or row['deadline_ms']!=m['deadline_ms']:raise ValueError('Control request binding differs')
            requests[meta['request_id']]=(c,m)
        elif row['kind']=='action_started':
            if requests[row['ticket_id']][1]['role']!='s1':raise ValueError('Non-action component applied')
        elif row['kind']=='action_finished' and row['status']=='completed':
            if row['pipeline']!=requests[row['ticket_id']][0]['audit']['pipeline']:raise ValueError('Action pipeline differs')
    if value!=json.loads((p/'result.json').read_text()):raise ValueError('Dynamic control summary differs from raw trace')
    return value
