"""Separate entry point for explicitly unprofiled prototype validation."""
import uuid
import ray
from ray import serve
from .prototype_plan import validate


def deploy(plan):
    validate(plan)
    if not ray.is_initialized():raise ValueError('Explicit existing cluster required')
    if not plan['single_gpu_colocation_extension']:
        from .fleet_service import _deploy_workers
        return _deploy_workers(plan)
    from .dedicated_service import DedicatedGroup,Transport
    resources=plan['physical_actor_resources'];available=ray.available_resources()
    if any(available.get(k,0)<resources[v] for k,v in [('GPU','num_gpus'),('CPU','num_cpus')]):
        raise ValueError('Insufficient resources for single physical prototype actor')
    options=dict(resources);env=plan['workers'][0]['backend'].get('runtime_env')
    if env:options['runtime_env']=env
    name='rosa-prototype-'+uuid.uuid4().hex[:12]
    transport=Transport({},[name],dict(mode='unprofiled_prototype_validation',comparison_eligible=False))
    try:
        handle=serve.run(DedicatedGroup.options(name='colocated-models',ray_actor_options=options).bind(plan['workers'],plan['plan_id']),name=name,route_prefix=None)
        transport.handles.update({w['worker_id']:handle for w in plan['workers']})
        return transport
    except BaseException:transport.close();raise
