"""One Ray Serve replica, one shared model, real tensor batching (CPU verified)."""
import asyncio
import time
import uuid
from dataclasses import asdict

from ray import serve
from .batching import SessionBook,combine_observations,split_actions,empty_response
from .dummy import DummyPolicy


class SlowDummyPolicy(DummyPolicy):
    def __init__(self,delay_s=0): self.delay_s=delay_s
    def get_action(self,obs):
        time.sleep(self.delay_s)
        return super().get_action(obs)


@serve.deployment(num_replicas=1,max_ongoing_requests=256,ray_actor_options={'num_cpus':1,'num_gpus':0})
class SharedModel:
    def __init__(self,max_batch_size=4,batch_wait_s=.01,dummy_delay_s=0,policy_factory=None):
        if type(max_batch_size) is not int or not 1<=max_batch_size<=1024 or not 0<=batch_wait_s<=1:
            raise ValueError('Invalid batching configuration')
        self.book=SessionBook()
        self.policy=SlowDummyPolicy(dummy_delay_s) if policy_factory is None else policy_factory()
        self.execution_kind='cpu_dummy' if policy_factory is None else 'external_policy_unverified'
        self._infer.set_max_batch_size(max_batch_size)
        self._infer.set_batch_wait_timeout_s(batch_wait_s)
        self.model_calls=0;self.batch_sizes=[];self.batches_inflight=0;self.max_batches_inflight=0
        self.finished_requests=0

    async def reset(self,run_id,robot_id,episode_id):
        return self.book.reset(run_id,robot_id,episode_id)

    async def predict(self,metadata,observation):
        ticket=self.book.accept(metadata,observation)
        try:
            return await self._infer({'ticket':ticket,'observation':observation})
        finally:
            self.book.finish(ticket)
            self.finished_requests+=1

    @serve.batch(max_batch_size=4,batch_wait_timeout_s=.01,max_concurrent_batches=1)
    async def _infer(self,items):
        outputs=[empty_response(item['ticket']) for item in items]
        active=[i for i,item in enumerate(items) if self.book.current(item['ticket'])]
        if not active:return outputs
        batched=combine_observations([items[i]['observation'] for i in active])
        batch_id=uuid.uuid4().hex
        begin=time.monotonic_ns()
        self.model_calls+=1;self.batch_sizes.append(len(active));self.batches_inflight+=1
        self.max_batches_inflight=max(self.max_batches_inflight,self.batches_inflight)
        try:
            action,_=await asyncio.to_thread(self.policy.get_action,batched)
            split=split_actions(action,len(active))
            end=time.monotonic_ns()
            for index,action in zip(active,split):
                ticket=items[index]['ticket']
                outputs[index].update(batch_id=batch_id,actual_batch_size=len(active),
                    server_timing=dict(queue_wait_ns=begin-ticket['enqueued_ns'],model_wall_ns=end-begin))
                if self.book.current(ticket): outputs[index].update(status='ok',action=action)
            return outputs
        finally:
            self.batches_inflight-=1

    async def stats(self):
        return dict(execution_kind=self.execution_kind,model_calls=self.model_calls,
            actual_batch_sizes=list(self.batch_sizes),max_batches_inflight=self.max_batches_inflight,
            inflight_batches=self.batches_inflight,finished_requests=self.finished_requests,
            busy_sessions=sum(s['busy'] for s in self.book.sessions.values()))
