"""Transactional global R07 replacement: plan, quiesce, release, install, rollback.

No overlapping generations or speculative actions. Physical state stays with the
robot/simulator; only model sessions are reset at an acknowledged boundary.
"""
import asyncio,json,time,uuid
from pathlib import Path
from copy import deepcopy
from .global_planner import validate_catalog,population,validate_plan
from .dynamic_control import Controller,prepare as control_spec,reaudit as control_audit
from .execution_plan import runtime_source_manifest,digest
from .regression import write_json,sha256
from .adapter import JsonlSink


def prepare(catalog,initial,gpu_budget,*,call_timeout_s=5.,deployment_timeout_s=600.):
    import math
    validate_catalog(catalog);initial=population(catalog,initial)
    if type(gpu_budget) is not int or not 0<=gpu_budget<=128:raise ValueError('Invalid physical GPU budget')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=3600 for v in (call_timeout_s,deployment_timeout_s)):raise ValueError('Bounded timeouts required')
    if call_timeout_s>600:raise ValueError('Model call timeout exceeds the component controller limit')
    v=dict(kind='global_control_spec',catalog=deepcopy(catalog),initial_population=initial,gpu_budget=gpu_budget,call_timeout_s=call_timeout_s,
        deployment_timeout_s=deployment_timeout_s,source_manifest=runtime_source_manifest(),transition_policy='quiesce, drain, release old leases, install; rollback only after confirmed cleanup',minimum_rate_scope='active classes only')
    v['id']=digest(v);return v


def validate(spec):
    if spec!=prepare(spec['catalog'],spec['initial_population'],spec['gpu_budget'],call_timeout_s=spec['call_timeout_s'],deployment_timeout_s=spec['deployment_timeout_s']):raise ValueError('Global spec/source changed')
    return spec


def check_decision(spec,decision,wanted,budget,revision):
    if decision['status'] not in ('planned','idle'):return
    if decision['population']!=wanted or decision['gpu_budget']!=budget or decision['revision']!=revision or decision['source_manifest']!=runtime_source_manifest():raise ValueError('Planner changed requested population/budget/source')
    p=decision['plan']
    if p is None:
        if any(wanted.values()) or decision['status']!='idle':raise ValueError('Nonempty plan missing')
        return
    validate_plan(p);x=p['inputs'];cat=spec['catalog'];active={c:rs for c,rs in wanted.items() if rs};fleet=deepcopy(cat['fleet'])
    fleet['classes']=[dict(c,robots=len(active[c['id']])) for c in fleet['classes'] if c['id'] in active];fleet['hardware']['gpu_count']=budget
    ids={m['id'] for c in fleet['classes'] for m in c['components']};fleet['profiles']=[p for p in fleet['profiles'] if p['component_id'] in ids]
    if x['fleet']!=fleet or x['roles']!={m:cat['roles'][m] for m in ids} or x['bindings']!={m:cat['bindings'][m] for m in ids} or x['mode']!=cat['mode'] or p['population_inputs']['robots']!=active or p['population_inputs']['revision']!=revision:raise ValueError('Candidate differs from catalog')


class Supervisor:
    def __init__(self,spec,manager,planner,output):
        validate(spec);self.spec=spec;self.manager=manager;self.planner=planner;self.out=Path(output);self.out.mkdir(parents=True,exist_ok=False)
        write_json(self.out/'spec.json',spec);self.sink=JsonlSink(self.out/'journal.jsonl');self.rows=[];self.run_id='dynamic-'+uuid.uuid4().hex
        self.lock=asyncio.Lock();self.gate=asyncio.Lock();self.calls=set();self.current=None;self.epoch=0;self.attempt=0;self.generations=[]
        self.population={c:[] for c in spec['initial_population']};self.admitted_since={};self.budget=spec['gpu_budget'];self.moving=False;self.blocked=False;self.closed=False
        self.emit('started',run_id=self.run_id,spec_id=spec['id'])
    def emit(self,kind,**kw):
        r=dict(kind=kind,time_ns=time.monotonic_ns(),**kw);self.rows.append(r);self.sink(r);return r
    def set_window(self,warmup_s,duration_s):
        import math
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=86400 for x in (warmup_s,duration_s)) or any(r['kind']=='measurement_window' for r in self.rows):raise ValueError('One future fixed window required')
        start=time.monotonic_ns()+round(warmup_s*1e9);end=start+round(duration_s*1e9)
        self.emit('measurement_window',window_start_ns=start,window_end_ns=end,run_epoch_ns=self.rows[0]['time_ns'])
        return start,end
    async def start(self):return await self.reconcile(self.spec['initial_population'],reason='initial')
    async def predict(self,rid,mid,observation,context=None,**timing):
        async with self.gate:
            if self.moving or self.blocked or self.closed or not self.current:return dict(status='blocked')
            c=self.current;epoch=self.epoch
            if rid not in c.active:return dict(status='not_active')
            admitted=self.admitted_since[rid]
            if timing.get('scheduled_ns') is not None and timing['scheduled_ns']<max(c.state[rid]['start_ns'],admitted):return dict(status='slot_before_epoch')
            t=asyncio.create_task(c.predict(rid,mid,observation,context,**timing));self.calls.add(t)
        try:
            r=await t
            if r.get('ticket_id'):r['ticket_id']=str(epoch)+':'+r['ticket_id']
            if r.get('request_id'):r['request_id']=str(epoch)+':'+r['request_id']
            return r
        finally:self.calls.discard(t)
    async def apply(self,ticket_id,executor):
        async with self.gate:
            if self.moving or self.blocked or self.closed or not self.current:return dict(status='blocked')
            parts=str(ticket_id).split(':',1)
            if len(parts)!=2 or parts[0]!=str(self.epoch):return dict(status='rejected_stale')
            t=asyncio.create_task(self.current.apply(parts[1],executor));self.calls.add(t)
        try:return await t
        finally:self.calls.discard(t)
    async def _quiesce(self,*,abandon_model_rpcs=False):
        if self.current:
            c=self.current
            async with c.lock:
                for r in c.active:
                    c.state[r]['epoch']+=1;c.state[r]['paused']=True;c.state[r]['plan']=None;c.state[r]['safety']=None
                c.emit('quiesced',robots=sorted(c.active))
            # A deleted Serve application's outstanding RPC can wait forever.
            # Model output is already invalidated by the epoch fence; physical
            # actions are a separate obligation and must actually finish.
            tasks=list(self.calls|set(c.actions.values())|(set() if abandon_model_rpcs else c.pending))
            if tasks:
                _,pending=await asyncio.wait(tasks,timeout=self.spec['call_timeout_s']+max(x['action_time_s'] for x in c.plan['classes']))
                if pending:self.emit('transition_blocked',reason='inflight_work_not_drained');return False
                await asyncio.gather(*tasks,return_exceptions=True)
            if any(s['unsafe'] for s in c.state.values()):self.emit('transition_blocked',reason='action_completion_unknown');return False
            if abandon_model_rpcs and c.pending:
                pending=list(c.pending);c.emit('invalidated_model_rpcs',count=len(pending),reason='failed_generation_released_before_replacement')
                for task in pending:task.cancel()
                await asyncio.gather(*pending,return_exceptions=True);c.pending.difference_update(pending)
        return True
    async def _release(self):
        if self.current and not self.current.closed:
            await self.current.close(reason='reconfigured');self.emit('generation_closed',epoch=self.epoch,directory=str(self.current.out.relative_to(self.out)))
        self.current=None
        await asyncio.wait_for(asyncio.shield(self.manager.release()),self.spec['deployment_timeout_s'])
        self.emit('leases_released',remaining_owned_leases=0)
    async def _install(self,plan,budget,pop):
        self.epoch+=1
        if plan is None:self.current=None;return
        spec=control_spec(plan,[r for rs in pop.values() for r in rs],{},physical_gpu_budget=budget,timeout_s=self.spec['call_timeout_s'])
        t=await asyncio.wait_for(asyncio.shield(self.manager.deploy(spec)),self.spec['deployment_timeout_s'])
        directory=self.out/f'generation-{self.epoch:04d}';c=Controller(spec,t,directory);self.current=c;self.generations.append(str(directory.relative_to(self.out)))
        await c.start()
        self.emit('generation_ready',epoch=self.epoch,plan_id=plan['plan_id'],directory=str(directory.relative_to(self.out)),physical_leases=len(plan['workers']))
    async def reconcile(self,wanted,*,gpu_budget=None,reason='population_change',forced=False,failed_worker=None):
        async with self.lock:
            if self.closed or self.blocked:return dict(status='blocked')
            if failed_worker is not None:
                if not self.current or failed_worker not in self.current.inventory:return dict(status='obsolete_failure')
                self.emit('failure_detected',worker_id=failed_worker,epoch=self.epoch)
            if forced:
                async with self.gate:self.moving=True
            if callable(wanted):wanted=wanted(deepcopy(self.population))
            if wanted is None:wanted=deepcopy(self.population)
            wanted=population(self.spec['catalog'],wanted);budget=self.budget if gpu_budget is None else gpu_budget;self.attempt+=1
            if type(budget) is not int or not 0<=budget<=128:raise ValueError('Invalid requested GPU budget')
            self.emit('plan_requested',attempt=self.attempt,population=wanted,gpu_budget=budget,reason=reason)
            try:
                d=await self.planner(self.spec['catalog'],wanted,budget,self.attempt,self.out/f'planning-{self.attempt:04d}')
                check_decision(self.spec,d,wanted,budget,self.attempt)
                if d['status'] not in ('planned','idle'):raise ValueError('No validated feasible witness')
                if d['plan']:await self.manager.preflight(d['plan'])
            except Exception as exc:
                self.emit('plan_rejected',attempt=self.attempt,error_type=type(exc).__name__,reason=str(exc))
                if forced or reason=='initial':
                    async with self.gate:self.moving=True;self.blocked=True
                    await self._quiesce()
                return dict(status='unresolved',old_population_preserved=not forced)
            write_json(self.out/f'decision-{self.attempt:04d}.json',d)
            async with self.gate:self.moving=True
            self.emit('transition_started',attempt=self.attempt,old_population=deepcopy(self.population),new_population=wanted,new_budget=budget)
            if not await self._quiesce(abandon_model_rpcs=forced):self.blocked=True;return dict(status='blocked_unknown_work')
            old_plan=self.current.plan if self.current else None;old_pop=deepcopy(self.population);old_budget=self.budget
            try:
                await self._release();await self._install(d['plan'],budget,wanted)
            except Exception as exc:
                self.emit('install_failed',error_type=type(exc).__name__,reason=str(exc))
                # Release is itself bounded; failure means uncertain ownership,
                # never permission to start another set of GPU workers.
                try:
                    await self._release()
                    if forced or (old_plan and len(old_plan['workers'])>budget):raise ValueError('Rollback incompatible with current resource limit')
                    await self._install(old_plan,min(old_budget,budget),old_pop)
                    self.population=old_pop;self.budget=min(old_budget,budget);self.moving=False
                    self._membership_commit('rollback_committed',old_pop,self.budget)
                    return dict(status='rolled_back')
                except Exception as rollback:
                    self.blocked=True;self.emit('transition_blocked',reason='cleanup_or_rollback_failed',error_type=type(rollback).__name__);return dict(status='blocked')
            self.population=wanted;self.budget=budget;self.moving=False
            self._membership_commit('population_committed',wanted,budget)
            return dict(status='committed',epoch=self.epoch,plan_id=d['plan']['plan_id'] if d['plan'] else None)
    def _membership_commit(self,kind,population,budget):
        row=self.emit(kind,population=population,gpu_budget=budget,epoch=self.epoch)
        previous=self.admitted_since
        self.admitted_since={rid:previous.get(rid,row['time_ns']) for rs in population.values() for rid in rs}
    async def join(self,class_id,robot_id):
        def change(current):
            if class_id not in current:raise ValueError('Unregistered class requires explicit catalog/profile revision')
            if any(robot_id in rs for rs in current.values()):raise ValueError('Robot already active')
            current[class_id].append(robot_id);return current
        return await self.reconcile(change,reason='join')
    async def leave(self,robot_id):
        def change(current):
            if not any(robot_id in rs for rs in current.values()):raise ValueError('Robot not active')
            return {c:[r for r in rs if r!=robot_id] for c,rs in current.items()}
        return await self.reconcile(change,reason='leave')
    async def recover_failure(self,worker_id,*,gpu_budget=None):
        return await self.reconcile(None,gpu_budget=gpu_budget,reason='worker_failure',forced=True,failed_worker=worker_id)
    async def close(self):
        async with self.lock:
            if self.closed:return
            async with self.gate:self.moving=True
            clean=await self._quiesce()
            try:await self._release();released=True
            except Exception as exc:self.emit('cleanup_failed',error_type=type(exc).__name__);released=False
            self.closed=True;self.emit('closed',clean=clean and released and not self.blocked,population=self.population,generations=self.generations)
            self.sink.close()
            from .global_audit import seal_tree
            write_json(self.out/'control_result.json',dict(status='complete' if clean and released and not self.blocked else 'incomplete',generations=self.generations,cleanup_confirmed=released))
            seal_tree(self.out)
