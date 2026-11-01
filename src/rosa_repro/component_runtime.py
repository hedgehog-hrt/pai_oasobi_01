"""Typed P1-P4 component protocol and worker core, independent of Ray.
Backends own model loading; the core owns sessions, batch attribution and reset.
"""
import asyncio
from copy import deepcopy
import importlib
import time
import uuid
import inspect
import threading
from . import gpu_measurement
from dataclasses import dataclass

from .adapter import ProtocolError,check_observation_batch_one,check_action_batch_one
from .batching import combine_observations,split_actions
from .dummy import DummyPolicy
from .execution_plan import digest,runtime_source_manifest

META_KEYS={'plan_id','worker_id','run_id','robot_id','episode_id','request_id','attempt_id','component_id','observation_seq','generation'}


class AdmissionRejected(ProtocolError):
    """The preceding timed-out invocation still owns this component session."""


@dataclass
class ComponentResult:
    """Per-request model output plus auditable generation/parser evidence."""
    output: dict | None
    evidence: dict
    error_type: str | None = None


def validate_output(role,value):
    if role=='s1':check_action_batch_one(value);return
    expected={'s2':{'plan'},'safety':{'stop','reason'},'monitor':{'text'}}[role]
    structured_monitor=role=='monitor' and isinstance(value,dict) and set(value)=={'text','task_status'}
    if not isinstance(value,dict) or (set(value)!=expected and not structured_monitor):raise ProtocolError('Component output schema mismatch')
    if structured_monitor and value['task_status'] not in ('ongoing','done','failed','unknown'):raise ProtocolError('Invalid monitor task status')
    if role=='safety' and type(value['stop']) is not bool:raise ProtocolError('Safety decision must be boolean')
    for k,v in value.items():
        if k!='stop' and (not isinstance(v,str) or not v.strip()):raise ProtocolError('Empty component text')


def s1_observation(payload,instruction_key='task'):
    obs=deepcopy(payload['observation']);check_observation_batch_one(obs)
    plan=payload.get('plan')
    if plan is not None:
        if not isinstance(plan,str) or not plan.strip():raise ProtocolError('Invalid S2 plan text')
        base=obs.get(instruction_key)
        if base is None:raise ProtocolError('S1 instruction key missing for plan injection')
        text=base if isinstance(base,str) else base[0]
        obs[instruction_key]=[text+'\nPlan: '+plan]
    return obs


class DummyBackend:
    def __init__(self,worker,delay_s=0,fail_calls=(),stop_calls=(),invalid_calls=()):
        if not 0<=delay_s<=5:raise ValueError('Invalid dummy delay')
        self.worker=worker;self.delay_s=delay_s;self.fail_calls=set(fail_calls);self.stop_calls=set(stop_calls)
        self.invalid_calls=set(invalid_calls);self.calls=0;self.plan_inputs=0
    def describe(self):
        return dict(execution_kind='cpu_dummy',stateless=True,component_id=self.worker['component_id'],role=self.worker['role'],
            model_revision=self.worker['model_revision'],runtime_fingerprint=self.worker['runtime_fingerprint'],
            model_artifacts_verified=False,device_type='cpu')
    def predict_batch(self,payloads):
        self.calls+=1;time.sleep(self.delay_s)
        if self.calls in self.fail_calls:raise RuntimeError('Injected dummy model failure')
        if self.calls in self.invalid_calls:return [None]*len(payloads)
        role=self.worker['role']
        if role=='s1':
            self.plan_inputs+=sum(p.get('plan') is not None for p in payloads)
            obs=combine_observations([s1_observation(p) for p in payloads])
            return split_actions(DummyPolicy().get_action(obs)[0],len(payloads))
        if role=='s2':return [dict(plan='Synthetic plan for '+str(float(p['observation']['state.x'][0,0,0]))) for p in payloads]
        if role=='safety':return [dict(stop=self.calls in self.stop_calls,reason='Synthetic safety decision') for p in payloads]
        return [dict(text='Synthetic monitor report') for p in payloads]


class AsyncDummyBackend(DummyBackend):
    """Labelled CPU harness for Serve's concurrent asynchronous backend path."""
    def __init__(self,worker,async_concurrency,**kwargs):
        super().__init__(worker,**kwargs)
        if worker['role']=='s1':raise ProtocolError('Async dummy only represents non-S1 generation engines')
        self.concurrency=async_concurrency;self.async_delay_s=self.delay_s;self.delay_s=0
    def describe(self):return dict(super().describe(),max_concurrent_requests=self.concurrency,serving_mode='synthetic_async_probe')
    async def predict_batch(self,payloads):
        await asyncio.sleep(self.async_delay_s)
        values=super().predict_batch(payloads)
        if self.worker['role']=='monitor':
            return [ComponentResult(dict(v,task_status='ongoing'),dict(raw_generation='synthetic monitor',
                                    evidence_kind='synthetic_async_probe',parse_status='ok')) for v in values]
        return values


def dummy_backend(worker,async_concurrency=1,**kwargs):
    if async_concurrency!=1:return AsyncDummyBackend(worker,async_concurrency,**kwargs)
    return DummyBackend(worker,**kwargs)


class ComponentCore:
    def __init__(self,worker,plan_id):
        self.worker=deepcopy(worker);self.plan_id=plan_id;self.source_sha256=runtime_source_manifest()['sha256']
        module,name=worker['backend']['factory'].split(':')
        kwargs=deepcopy(worker['backend']['kwargs']);enabled=kwargs.pop('cuda_measurement',False)
        self.backend=getattr(importlib.import_module(module),name)(worker=deepcopy(worker),**kwargs)
        self.description=self.backend.describe()
        self.measurement=gpu_measurement.contract(enabled,gpu=bool(worker['resources']['num_gpus']),asynchronous=inspect.iscoroutinefunction(self.backend.predict_batch))
        self.description['gpu_measurement']=self.measurement
        self.measurement_lock=threading.Lock()
        self.max_concurrent_batches=self.description.get('max_concurrent_requests',1)
        if type(self.max_concurrent_batches) is not int or not 1<=self.max_concurrent_batches<=128:
            raise ProtocolError('Invalid backend concurrency')
        if self.max_concurrent_batches>1 and (not inspect.iscoroutinefunction(self.backend.predict_batch) or worker['role']=='s1' or worker['batch_size']!=1 or worker['batch_wait_ms']!=0):
            raise ProtocolError('Continuous engine requires async non-S1, batch-one admission and zero batch wait')
        for key in ('component_id','role','model_revision','runtime_fingerprint'):
            if self.description.get(key)!=worker[key]:raise ProtocolError('Backend identity mismatch: '+key)
        if self.description.get('stateless') is not True:raise ProtocolError('Stateful backend needs a separate session contract')
        expected='external_model' if worker['resources']['num_gpus'] else 'cpu_dummy'
        if self.description.get('execution_kind')!=expected:raise ProtocolError('Backend execution kind mismatch')
        self.sessions={};self.episodes=set();self.seen=set();self.instance_id=uuid.uuid4().hex
        self.model_calls=0;self.batch_sizes=[];self.inflight_batches=0;self.max_inflight_batches=0
    def reset(self,run_id,robot_id,episode_id):
        if any(not isinstance(v,str) or not v for v in (run_id,robot_id,episode_id)):raise ProtocolError('Invalid reset identity')
        if robot_id not in self.worker['members']:raise ProtocolError('Robot routed to wrong worker')
        if (run_id,robot_id,episode_id) in self.episodes:raise ProtocolError('Duplicate episode')
        key=(run_id,robot_id);generation=self.sessions[key]['generation']+1 if key in self.sessions else 0
        self.sessions[key]=dict(episode_id=episode_id,generation=generation,last_seq=-1,busy=False)
        self.episodes.add((run_id,robot_id,episode_id))
        return dict(run_id=run_id,robot_id=robot_id,episode_id=episode_id,generation=generation,
                    worker_id=self.worker['worker_id'],plan_id=self.plan_id,instance_id=self.instance_id)
    def accept(self,meta,payload):
        if set(meta)!=META_KEYS:raise ProtocolError('Invalid component metadata fields')
        if any(not isinstance(meta[k],str) or not meta[k] for k in META_KEYS-{'observation_seq','generation'}):
            raise ProtocolError('Invalid component identity')
        if any(type(meta[k]) is not int or meta[k]<0 for k in ('observation_seq','generation')):raise ProtocolError('Invalid sequence/generation')
        if any(meta[k]!=v for k,v in dict(plan_id=self.plan_id,worker_id=self.worker['worker_id'],component_id=self.worker['component_id']).items()):
            raise ProtocolError('Wrong plan/component/worker route')
        check_observation_batch_one(payload['observation'])
        key=(meta['run_id'],meta['robot_id']);state=self.sessions.get(key)
        if state is None or state['episode_id']!=meta['episode_id'] or state['generation']!=meta['generation']:
            raise ProtocolError('Stale/unknown session')
        if (meta['run_id'],meta['request_id']) in self.seen:raise ProtocolError('Duplicate request')
        if meta['observation_seq']<=state['last_seq']:raise ProtocolError('Out-of-order component session')
        if state['busy']:raise AdmissionRejected('Component session still busy')
        state['busy']=True;state['last_seq']=meta['observation_seq'];self.seen.add((meta['run_id'],meta['request_id']))
        return dict(meta=deepcopy(meta),payload=deepcopy(payload),state=state,enqueued_ns=time.monotonic_ns())
    def current(self,t):return self.sessions[(t['meta']['run_id'],t['meta']['robot_id'])] is t['state']
    def finish(self,t):t['state']['busy']=False
    def response(self,t):
        return dict(metadata=deepcopy(t['meta']),status='cancelled',output=None,batch_id=None,actual_batch_size=None,
                    instance_id=self.instance_id,worker_id=self.worker['worker_id'],plan_id=self.plan_id)
    async def run_batch(self,tickets):
        outputs=[self.response(t) for t in tickets];active=[i for i,t in enumerate(tickets) if self.current(t)]
        if not active:return outputs
        begin=time.monotonic_ns();batch_id=uuid.uuid4().hex
        self.model_calls+=1;self.batch_sizes.append(len(active));self.inflight_batches+=1
        self.max_inflight_batches=max(self.max_inflight_batches,self.inflight_batches)
        measurement=None
        try:
            payloads=[tickets[i]['payload'] for i in active]
            if inspect.iscoroutinefunction(self.backend.predict_batch):
                measurement=gpu_measurement.unavailable(self.measurement['unavailable_reason'])
                values=await self.backend.predict_batch(payloads)
            else:
                def measured_call():
                    with self.measurement_lock:
                        return gpu_measurement.invoke(lambda:self.backend.predict_batch(payloads),self.measurement)
                values,measurement,exception=await asyncio.to_thread(measured_call)
                if exception:raise exception
            if not isinstance(values,list) or len(values)!=len(active):raise ProtocolError('Backend batch output length mismatch')
            for value in values:
                if isinstance(value,ComponentResult):
                    if value.error_type is None:validate_output(self.worker['role'],value.output)
                    elif value.output is not None:raise ProtocolError('Failed generation must not contain executable output')
                else:validate_output(self.worker['role'],value)
            status='ok';error=None
        except Exception as exc:
            values=[None]*len(active);status='error';error=type(exc).__name__
        finally:self.inflight_batches-=1
        end=time.monotonic_ns()
        for i,value in zip(active,values):
            t=tickets[i];outputs[i].update(batch_id=batch_id,actual_batch_size=len(active),
                server_timing=dict(queue_wait_ns=begin-t['enqueued_ns'],model_wall_ns=end-begin),error_type=error)
            item_status=status
            if isinstance(value,ComponentResult):
                outputs[i].update(backend_evidence=value.evidence,error_type=value.error_type)
                item_status='timeout' if value.error_type=='ModelGenerationTimeout' else 'error' if value.error_type else status
                value=value.output
            if measurement is not None:
                evidence=outputs[i].setdefault('backend_evidence',{})
                evidence['gpu_measurement']=dict(measurement,batch_id=batch_id,actual_batch_size=len(active))
            if self.current(t):outputs[i].update(status=item_status,output=value)
        return outputs
    def stats(self):
        return dict(plan_id=self.plan_id,worker_spec_sha256=digest(self.worker),instance_id=self.instance_id,
            execution_kind=self.description['execution_kind'],backend_attestation=self.description,
            runtime_source_sha256=self.source_sha256,
            model_calls=self.model_calls,actual_batch_sizes=list(self.batch_sizes),inflight_batches=self.inflight_batches,
            model_call_counter_semantics='backend batch invocations; not vLLM internal token batches',
            max_inflight_batches=self.max_inflight_batches,busy_sessions=sum(s['busy'] for s in self.sessions.values()),
            configured_concurrency=self.max_concurrent_batches,
            backend_plan_inputs=getattr(self.backend,'plan_inputs',None))
