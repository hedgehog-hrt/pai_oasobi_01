"""Ray Serve binding for a static, validated R07 execution plan.
Each worker is a separate deployment/replica. GPU mode reserves one full GPU.
"""
import asyncio
import os
import sys
import uuid
import ray
from ray import serve

from .component_runtime import ComponentCore,AdmissionRejected
from .execution_plan import validate_execution_plan,check_runtime_inventory,check_gpu_device


@serve.deployment(num_replicas=1,max_ongoing_requests=256)
class ComponentWorker:
    def __init__(self,worker,plan_id):
        self.gpu_properties=None
        if worker['resources']['num_gpus']:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
                raise ValueError('Expected one allocated and visible CUDA device')
            props=torch.cuda.get_device_properties(0)
            self.gpu_properties=dict(name=props.name,memory_mib=props.total_memory/1024**2,uuid=str(getattr(props,'uuid','unknown')))
            check_gpu_device(worker,ray.get_runtime_context().get_accelerator_ids().get('GPU',[]),self.gpu_properties,
                             cuda_available=torch.cuda.is_available(),visible_count=torch.cuda.device_count())
        self.core=ComponentCore(worker,plan_id);self.pending=set()
        self._infer.set_max_batch_size(worker['batch_size'])
        self._infer.set_batch_wait_timeout_s(worker['batch_wait_ms']/1000)
        self.continuous_gate=asyncio.Semaphore(self.core.max_concurrent_batches)
    async def reset(self,*args):return self.core.reset(*args)
    async def predict(self,metadata,payload):
        try:ticket=self.core.accept(metadata,payload)
        except AdmissionRejected:return dict(self.core.response({'meta':metadata}),status='rejected')
        async def dispatch():
            try:
                if self.core.max_concurrent_batches>1:
                    # A continuous engine receives individual requests directly;
                    # its own scheduler owns token-level batches.
                    async with self.continuous_gate:return (await self.core.run_batch([ticket]))[0]
                return await self._infer(ticket)
            finally:self.core.finish(ticket)
        task=asyncio.create_task(dispatch());self.pending.add(task)
        task.add_done_callback(self.pending.discard)
        # A client timeout must not free the model/session while work is running.
        return await asyncio.shield(task)
    @serve.batch(max_batch_size=1,batch_wait_timeout_s=0,max_concurrent_batches=1)
    async def _infer(self,tickets):return await self.core.run_batch(tickets)
    async def stats(self):
        ctx=ray.get_runtime_context()
        return dict(self.core.stats(),node_id=ctx.get_node_id(),pid=os.getpid(),python_executable=sys.executable,
                    gpu_ids=[str(x) for x in ctx.get_accelerator_ids().get('GPU',[])],
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_properties=self.gpu_properties)
    async def profile_telemetry(self):
        from .profile_protocol import gpu_telemetry
        return await asyncio.to_thread(gpu_telemetry,self.gpu_properties)
    async def profile_environment(self):
        from .profile_protocol import environment_evidence
        return await asyncio.to_thread(environment_evidence)

    async def simulation_interface(self):
        if self.core.worker['role']!='s1' or not hasattr(self.core.backend,'policy'):raise ValueError('Loaded S1 policy required')
        config=self.core.backend.policy.get_modality_config()
        return dict(plan_id=self.core.plan_id,worker_id=self.core.worker['worker_id'],instance_id=self.core.instance_id,
            modality_config={k:dict(delta_indices=list(v.delta_indices),modality_keys=list(v.modality_keys)) for k,v in config.items()})


class RayFleetTransport:
    def __init__(self,handles,application_names):self.handles=handles;self.application_names=application_names
    async def reset(self,wid,*args):return await self.handles[wid].reset.remote(*args)
    async def predict(self,wid,*args):return await self.handles[wid].predict.remote(*args)
    async def worker_stats(self,wid):return await self.handles[wid].stats.remote()
    def fail_worker(self,wid):
        index=list(self.handles).index(wid);serve.delete(self.application_names[index])
    async def inventory(self):
        values=await asyncio.gather(*(h.stats.remote() for h in self.handles.values()))
        return dict(zip(self.handles,values))
    async def telemetry(self,wid):return await self.handles[wid].profile_telemetry.remote()
    async def environment(self,wid):return await self.handles[wid].profile_environment.remote()
    async def simulation_interface(self,wid):return await self.handles[wid].simulation_interface.remote()
    def close(self):
        for name in self.application_names:serve.delete(name)
        self.handles.clear();self.application_names.clear()


def deploy_fleet(plan,*,adoptions=None,residency_calibration=False,joint_review=None):
    """Call on an already initialized Ray cluster. Does not rent/start a Pod."""
    validate_execution_plan(plan)
    if plan.get('purpose')=='single_gpu_prototype_validation':
        raise ValueError('Use the separate prototype validation entry point; this is not reviewed R07 deployment')
    if plan.get('purpose')=='dedicated_GPU_execution':
        from .dedicated_service import deploy
        return deploy(plan,adoptions=adoptions,calibration=residency_calibration,joint_review=joint_review)
    if residency_calibration or joint_review is not None:raise ValueError('Dedicated residency options require a dedicated plan')
    from .profile_adoption import validate_fleet_adoptions,runtime_signature
    accepted=validate_fleet_adoptions(plan,adoptions)
    transport=_deploy_workers(plan)
    try:
        for w in plan['workers'] if plan['mode']=='gpu' else []:
            handle=transport.handles[w['worker_id']]
            env,stats=ray.get([handle.profile_environment.remote(),handle.stats.remote()],timeout=60)
            if runtime_signature(env,stats['backend_attestation'])!=accepted[w['profile_id']]['runtime_signature']:
                raise ValueError('Deployed worker runtime differs from reviewed profile')
        return transport
    except BaseException:
        transport.close();raise


def deploy_probe(plan):
    """An isolated measurement worker; no fabricated R07 profile is needed."""
    from .profile_protocol import validate_probe_plan
    validate_probe_plan(plan)
    return _deploy_workers(plan)


def deploy_r05(plan):
    """Isolated S1 replay measurement with no scheduling or latency-SLO contract."""
    from .r05_protocol import validate_plan
    validate_plan(plan)
    return _deploy_workers(plan)


def _deploy_workers(plan):
    if plan.get('purpose')=='dedicated_GPU_execution':raise ValueError('Dedicated groups require their physical-group executor')
    if not ray.is_initialized():raise RuntimeError('Initialize the intended Ray cluster explicitly')
    need=sum(w['resources']['num_gpus'] for w in plan['workers'])
    available=ray.available_resources()
    if available.get('GPU',0)<need:raise ValueError('Insufficient available GPU resources; no CPU fallback')
    if available.get('CPU',0)<len(plan['workers']):raise ValueError('Insufficient available CPU resources')
    names=[];handles={};prefix='rosa-fleet-'+uuid.uuid4().hex[:12]
    try:
        for index,w in enumerate(plan['workers']):
            name=prefix+'-'+str(index);names.append(name)
            actor_options=dict(w['resources'])
            if 'runtime_env' in w['backend']:actor_options['runtime_env']=w['backend']['runtime_env']
            handles[w['worker_id']]=serve.run(ComponentWorker.options(name='component-'+str(index),
                ray_actor_options=actor_options).bind(w,plan['plan_id']),name=name,route_prefix=None)
        return RayFleetTransport(handles,names)
    except BaseException:
        for name in names:serve.delete(name)
        raise


def deploy_control(spec,*,adoptions=None):
    from .dynamic_control import validate,envelope
    from .profile_adoption import validate_fleet_adoptions,runtime_signature
    validate(spec);accepted=validate_fleet_adoptions(spec['blueprint'],adoptions)
    transport=_deploy_workers(envelope(spec))
    try:
        for w in spec['workers'] if spec['blueprint']['mode']=='gpu' else []:
            handle=transport.handles[w['worker_id']]
            env,stats=ray.get([handle.profile_environment.remote(),handle.stats.remote()],timeout=60)
            if runtime_signature(env,stats['backend_attestation'])!=accepted[w['profile_id']]['runtime_signature']:raise ValueError('Control worker runtime differs from adoption')
        return transport
    except BaseException:transport.close();raise


def deploy_r05_curve(plan):
    from .r05_load_curves import validate_plan
    validate_plan(plan);return _deploy_workers(plan)
