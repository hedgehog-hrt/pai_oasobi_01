"""One process owns a physical model group; FIFO by default, explicit prototypes optional."""
import asyncio
import heapq
import time
from collections import deque
from .component_runtime import ComponentCore,AdmissionRejected

PARALLEL_POLICIES={'parallel_roles_v1','parallel_roles_batch_s1_v1'}

class DeadlineGate:
    """Non-preemptive earliest-deadline selection; equal deadlines keep FIFO."""
    def __init__(self):self.busy=False;self.waiting=[];self.serial=0
    async def acquire(self,deadline):
        if not self.busy:self.busy=True;return
        future=asyncio.get_running_loop().create_future();self.serial+=1
        heapq.heappush(self.waiting,(deadline,self.serial,future))
        try:await future
        except BaseException:
            if future.done() and not future.cancelled():self.release()
            raise
    def release(self):
        while self.waiting:
            _,_,future=heapq.heappop(self.waiting)
            if not future.done():future.set_result(None);return
        self.busy=False


class GroupCore:
    def __init__(self,workers,plan_id):
        policies={w.get('group_scheduling',{}).get('policy','fifo') for w in workers}
        if len(policies)!=1 or not policies<=({'fifo','edf_queue_expiry_v1'}|PARALLEL_POLICIES):raise ValueError('Inconsistent group scheduling policy')
        self.policy=policies.pop();self.deadline_gate=DeadlineGate()
        self.expired={w['worker_id']:0 for w in workers}
        if self.policy=='edf_queue_expiry_v1':
            for w in workers:
                value=w['group_scheduling']['queue_budget_ms']
                if type(value) not in (int,float) or not 0<value<=60000:raise ValueError('Invalid queue budget')
        self.cores={w['worker_id']:ComponentCore(w,plan_id) for w in workers}
        self.gate=asyncio.Lock();self.pending=set();self.active=0;self.peak=0
        if self.policy in PARALLEL_POLICIES and len({w['role'] for w in workers})!=len(workers):raise ValueError('Parallel roles require one model per role')
        self.role_limits={}
        for wid,core in self.cores.items():
            limit=core.worker.get('group_scheduling',{}).get('max_concurrent_invocations',1)
            if type(limit) is not int or not 1<=limit<=min(4,core.max_concurrent_batches):raise ValueError('Role concurrency exceeds backend contract')
            if limit>1 and (self.policy!='parallel_roles_batch_s1_v1' or core.worker['role']=='s1'):raise ValueError('Concurrent role requires async VLM and parallel policy')
            self.role_limits[wid]=limit
        self.role_gates={wid:asyncio.Semaphore(limit) for wid,limit in self.role_limits.items()}
        self.batch_queues={wid:deque() for wid in self.cores};self.batch_pumps={};self.batch_arrivals={wid:asyncio.Event() for wid in self.cores}
    def reset(self,wid,*args):return self.cores[wid].reset(*args)
    async def predict(self,wid,metadata,payload):
        core=self.cores[wid]
        try:ticket=core.accept(metadata,payload)
        except AdmissionRejected:return dict(core.response({'meta':metadata}),status='rejected')
        if self.policy=='parallel_roles_batch_s1_v1' and core.worker['role']=='s1':
            future=asyncio.get_running_loop().create_future()
            self.batch_queues[wid].append((ticket,future));self.batch_arrivals[wid].set()
            if wid not in self.batch_pumps:
                task=asyncio.create_task(self._batch_pump(wid));self.batch_pumps[wid]=task
                self.pending.add(task);task.add_done_callback(self.pending.discard)
            return await asyncio.shield(future)
        async def work():
            deadline=ticket['enqueued_ns']+round(core.worker.get('group_scheduling',{}).get('queue_budget_ms',0)*1e6)
            acquired=False;gate=self.role_gates[wid] if self.policy in PARALLEL_POLICIES else self.gate
            try:
                if self.policy=='edf_queue_expiry_v1':await self.deadline_gate.acquire(deadline)
                else:await gate.acquire()
                acquired=True;dispatched=time.monotonic_ns()
                expired=self.policy=='edf_queue_expiry_v1' and dispatched>=deadline
                scheduling=dict(policy=self.policy,queue_wait_ms=(dispatched-ticket['enqueued_ns'])/1e6,
                    queue_deadline_ns=deadline if self.policy=='edf_queue_expiry_v1' else None,expired_before_inference=expired,
                    clock='physical-actor-monotonic',deadline_scope='worker admission plus role budget; end-to-end SLO evaluated separately',
                    dispatch_ns=dispatched,other_active_invocations_at_dispatch=self.active,
                    concurrency_scope=('bounded async within role' if self.role_limits[wid]>1 else 'one invocation per role') if self.policy in PARALLEL_POLICIES else 'one invocation per physical group',
                    max_concurrent_invocations=self.role_limits[wid] if self.policy in PARALLEL_POLICIES else 1,preemption=False)
                if expired:
                    self.expired[wid]+=1
                    result=dict(core.response(ticket),status='error',error_type='QueueDeadlineExceeded')
                else:
                    self.active+=1;self.peak=max(self.peak,self.active)
                    try:result=(await core.run_batch([ticket]))[0]
                    finally:self.active-=1
                scheduling['completed_ns']=time.monotonic_ns();result['scheduling_evidence']=scheduling
                return result
            finally:
                if acquired:
                    if self.policy=='edf_queue_expiry_v1':self.deadline_gate.release()
                    else:gate.release()
                core.finish(ticket)
        task=asyncio.create_task(work());self.pending.add(task);task.add_done_callback(self.pending.discard)
        return await asyncio.shield(task)
    async def _batch_pump(self,wid):
        """Own all accepted S1 tickets through cancellation, reset and failures.

        The first admission bounds batch assembly wait. Queue time during an
        earlier inference is included; requests never get a fresh wait budget.
        """
        core=self.cores[wid];queue=self.batch_queues[wid];arrived=self.batch_arrivals[wid]
        limit=core.worker['batch_size'];wait_ns=round(core.worker['batch_wait_ms']*1e6)
        try:
            while queue:
                cutoff=queue[0][0]['enqueued_ns']+wait_ns
                while len(queue)<limit and time.monotonic_ns()<cutoff:
                    arrived.clear()
                    try:await asyncio.wait_for(arrived.wait(),max(0,(cutoff-time.monotonic_ns())/1e9))
                    except TimeoutError:break
                batch=[queue.popleft() for _ in range(min(limit,len(queue)))];dispatched=time.monotonic_ns();others=self.active
                self.active+=1;self.peak=max(self.peak,self.active)
                try:
                    results=await core.run_batch([t for t,_ in batch]);completed=time.monotonic_ns()
                    for (ticket,future),result in zip(batch,results):
                        result['scheduling_evidence']=dict(policy=self.policy,queue_wait_ms=(dispatched-ticket['enqueued_ns'])/1e6,
                            queue_deadline_ns=None,expired_before_inference=False,clock='physical-actor-monotonic',
                            deadline_scope='end-to-end SLO evaluated separately; batch assembly wait included',
                            dispatch_ns=dispatched,completed_ns=completed,other_active_invocations_at_dispatch=others,
                            concurrency_scope='one invocation per role',preemption=False,
                            configured_batch_size=limit,configured_batch_wait_ms=core.worker['batch_wait_ms'])
                        if not future.done():future.set_result(result)
                except BaseException as exc:
                    for _,future in batch:
                        if not future.done():future.set_exception(exc)
                    raise
                finally:
                    self.active-=1
                    for ticket,_ in batch:core.finish(ticket)
        finally:
            self.batch_pumps.pop(wid,None)
            while queue:
                ticket,future=queue.popleft();core.finish(ticket)
                if not future.done():future.set_result(dict(core.response(ticket),status='error',error_type='BatchPumpStopped'))
    async def drain(self):
        if self.pending:await asyncio.gather(*self.pending)
    def stats(self,wid):return dict(self.cores[wid].stats(),physical_group_id=self.cores[wid].worker['physical_group_id'],
        group_peak_invocations=self.peak,group_queue_policy=self.policy,queue_expired_requests=self.expired[wid])
