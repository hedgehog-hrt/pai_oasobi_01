"""Ray Serve actor per physical dedicated GPU group; no fractional GPU allocation."""
import asyncio,os,sys,uuid
import ray
from ray import serve
from .dedicated_core import GroupCore
from .dedicated_execution import launch_gate
from .execution_plan import check_gpu_device


@serve.deployment(num_replicas=1,max_ongoing_requests=256)
class DedicatedGroup:
    def __init__(self,workers,plan_id):
        self.properties=None
        if workers[0]['resources']['num_gpus']:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One visible CUDA device required')
            p=torch.cuda.get_device_properties(0);self.properties=dict(name=p.name,memory_mib=p.total_memory/1024**2,uuid=str(getattr(p,'uuid','unknown')))
            for w in workers:check_gpu_device(w,ray.get_runtime_context().get_accelerator_ids().get('GPU',[]),self.properties,cuda_available=True,visible_count=1)
        self.group=GroupCore(workers,plan_id)
        from .gpu_telemetry_stream import GPUStream
        self.gpu_stream=GPUStream(self.properties)
    async def reset(self,wid,*args):return self.group.reset(wid,*args)
    async def predict(self,wid,*args):return await self.group.predict(wid,*args)
    async def stats(self,wid):
        c=ray.get_runtime_context()
        return dict(self.group.stats(wid),node_id=c.get_node_id(),pid=os.getpid(),python_executable=sys.executable,
            gpu_ids=[str(x) for x in c.get_accelerator_ids().get('GPU',[])],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_properties=self.properties)
    async def environment(self):
        from .profile_protocol import environment_evidence
        return await asyncio.to_thread(environment_evidence)
    async def telemetry(self):
        return await asyncio.to_thread(self.gpu_stream.sample)
    async def stop_telemetry(self):
        await asyncio.to_thread(self.gpu_stream.close)
    def __del__(self):
        if hasattr(self,'gpu_stream'):self.gpu_stream.close()
    async def simulation_interface(self,wid):
        core=self.group.cores[wid]
        if core.worker['role']!='s1' or not hasattr(core.backend,'policy'):raise ValueError('Loaded S1 policy required')
        config=core.backend.policy.get_modality_config()
        return dict(plan_id=core.plan_id,worker_id=wid,instance_id=core.instance_id,
            modality_config={k:dict(delta_indices=list(v.delta_indices),modality_keys=list(v.modality_keys)) for k,v in config.items()})


class Transport:
    def __init__(self,handles,names,gate):self.handles=handles;self.names=names;self.launch_gate=gate
    async def reset(self,wid,*args):return await self.handles[wid].reset.remote(wid,*args)
    async def predict(self,wid,*args):return await self.handles[wid].predict.remote(wid,*args)
    async def worker_stats(self,wid):return await self.handles[wid].stats.remote(wid)
    async def inventory(self):return dict(zip(self.handles,await asyncio.gather(*(h.stats.remote(w) for w,h in self.handles.items()))))
    async def environment(self,wid):return await self.handles[wid].environment.remote()
    async def telemetry(self,wid):return await self.handles[wid].telemetry.remote()
    async def simulation_interface(self,wid):return await self.handles[wid].simulation_interface.remote(wid)
    async def aclose(self):
        unique={id(h):h for h in self.handles.values()}
        await asyncio.gather(*(h.stop_telemetry.remote() for h in unique.values()))
        self.close()
    def close(self):
        for name in self.names:serve.delete(name)
        self.handles.clear();self.names.clear()


def deploy(plan,*,adoptions=None,calibration=False,joint_review=None):
    from .profile_adoption import validate_fleet_adoptions,runtime_signature
    gate=launch_gate(plan,calibration=calibration,joint_review=joint_review);accepted=validate_fleet_adoptions(plan,adoptions)
    if not ray.is_initialized():raise ValueError('Explicit existing cluster required')
    groups=plan['physical_groups'];available=ray.available_resources()
    if available.get('GPU',0)<sum(g['resources']['num_gpus'] for g in groups) or available.get('CPU',0)<len(groups):raise ValueError('Insufficient dedicated group resources; no partial workload or CPU fallback')
    names=[];handles={};workers={w['worker_id']:w for w in plan['workers']};prefix='rosa-dedicated-'+uuid.uuid4().hex[:12]
    t=Transport(handles,names,gate)
    try:
        for i,g in enumerate(groups):
            name=prefix+'-'+str(i);names.append(name);options=dict(g['resources'])
            if g['runtime_env']:options['runtime_env']=g['runtime_env']
            h=serve.run(DedicatedGroup.options(name='group-'+str(i),ray_actor_options=options).bind([workers[w] for w in g['worker_ids']],plan['plan_id']),name=name,route_prefix=None)
            for wid in g['worker_ids']:handles[wid]=h
        for w in plan['workers'] if plan['mode']=='gpu' else []:
            h=handles[w['worker_id']];env,stats=ray.get([h.environment.remote(),h.stats.remote(w['worker_id'])],timeout=60)
            if runtime_signature(env,stats['backend_attestation'])!=accepted[w['profile_id']]['runtime_signature']:raise ValueError('Dedicated runtime differs from adopted profile')
        return t
    except BaseException:t.close();raise
