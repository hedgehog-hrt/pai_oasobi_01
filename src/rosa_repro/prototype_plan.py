"""Explicit, unprofiled single-GPU connection validation, never an R07 adoption.

P1 keeps one model per GPU. Multi-model pipelines are a declared project
extension: one GPU actor and shared model instances, serialized by default.
"""
from copy import deepcopy
import math
from .execution_plan import digest,runtime_source_manifest,global_robot,check_gpu_device
from .profile_protocol import validate_probe_config
from .multicomponent import ROLES,paper_class

PURPOSE='single_gpu_prototype_validation'


def prepare(configs,population,*,frequency_hz=2.,action_time_s=.4,s1_headroom_fraction=.15,queue_policy='fifo',periodic_phases='synchronized',collision_guard=None,recovery_policy=None,role_concurrency=None,prefetch_resets=False):
    if queue_policy not in ('fifo','edf_queue_expiry_v1','parallel_roles_v1','parallel_roles_batch_s1_v1'):raise ValueError('Unknown prototype queue policy')
    if periodic_phases not in ('synchronized','spread_classes_v1'):raise ValueError('Unknown periodic phase policy')
    if not population or set(population)-set(ROLES) or any(type(n) is not int or not 1<=n<=16 for n in population.values()):
        raise ValueError('Explicit P1-P4 robot counts in 1..16 required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (frequency_hz,action_time_s)):
        raise ValueError('Finite positive frequency and action duration required')
    if not .05<=s1_headroom_fraction<=.5:raise ValueError('Explicit S1/headroom allowance required')
    needed=set().union(*(ROLES[p] for p in population))
    if set(configs)!=needed:raise ValueError('Exactly the required component configs must be supplied')
    role_concurrency=role_concurrency or {}
    if not isinstance(role_concurrency,dict) or set(role_concurrency)-needed or 's1' in role_concurrency:
        raise ValueError('Only required asynchronous VLM roles may have concurrency limits')
    if role_concurrency and queue_policy!='parallel_roles_batch_s1_v1':raise ValueError('Role concurrency requires explicit parallel S1 batching')
    for role,limit in role_concurrency.items():
        if type(limit) is not int or not 1<=limit<=4:raise ValueError('Role concurrency must be bounded to 1..4')
    if type(prefetch_resets) is not bool:raise ValueError('Reset prefetch must be explicit boolean')
    modes=set();hardware=set();envs=[]
    for role,c in configs.items():
        validate_probe_config(c)
        if c['role']!=role or c['component_id']!=role:raise ValueError('Canonical role/component binding required')
        modes.add(c['mode']);hardware.add(c['hardware_kind']);envs.append(c['binding'].get('runtime_env'))
    if len(modes)!=1 or len(hardware)!=1 or any(e!=envs[0] for e in envs):raise ValueError('Common hardware, mode and Python required')
    mode=next(iter(modes));extension=needed!={'s1'}
    visual_guard=configs.get('safety',{}).get('binding',{}).get('kwargs',{}).get('config',{}).get('structured_output')=='safety_visual_guard_v1'
    if visual_guard!=(collision_guard is not None):raise ValueError('Visual guard and independent geometry guard must be paired')
    if collision_guard is not None:
        from .hybrid_safety import validate_config as validate_guard
        validate_guard(collision_guard)
        if mode!='gpu' or 'safety' not in needed:raise ValueError('Hybrid guard requires real GPU visual Safety and LIBERO simulation')
    if recovery_policy is not None:
        from .recovery_policy import validate_config as validate_recovery
        validate_recovery(recovery_policy)
        if collision_guard is None:raise ValueError('Retreat requires the visual and geometry guards')
    if not extension and queue_policy!='fifo':raise ValueError('Group queue policy applies only to the colocation extension')
    if not extension and periodic_phases!='synchronized':raise ValueError('Periodic phases require the colocation extension')
    if queue_policy=='parallel_roles_batch_s1_v1':
        if configs['s1']['batch_size']>16 or configs['s1']['batch_wait_ms']>10:raise ValueError('S1 group batching is bounded to 16 requests and 10 ms')
        if any(c['batch_size']!=1 or c['batch_wait_ms']!=0 for r,c in configs.items() if r!='s1'):raise ValueError('VLM roles require batch one')
    elif extension and any(c['batch_size']!=1 or c['batch_wait_ms']!=0 for c in configs.values()):
        raise ValueError('Colocation prototype explicitly supports batch one only')
    fractions=[c['binding']['kwargs'].get('config',{}).get('gpu_memory_utilization',0) for r,c in configs.items() if r!='s1']
    if sum(fractions)+s1_headroom_fraction>.95:raise ValueError('VLM reservations leave insufficient S1/headroom allowance')
    classes=[];members={r:[] for r in needed}
    for pipeline,n in sorted(population.items()):
        robots=[]
        for i in range(n):
            rid=global_robot(pipeline,str(i));routes={r:'prototype-'+r for r in sorted(ROLES[pipeline])}
            robots.append(dict(robot_id=rid,local_robot_id=str(i),routes=routes))
            for r in routes:members[r].append(rid)
        classes.append(dict(audit=paper_class(pipeline,pipeline),frequency_hz=frequency_hz,action_time_s=action_time_s,robots=robots))
    if periodic_phases=='spread_classes_v1':
        for role in ('safety','monitor'):
            components=[m for c in classes for m in c['audit']['components'] if m['role']==role]
            for i,m in enumerate(components):m['phase_ns']=i*m['period_ns']//len(components)
    workers=[]
    for role in ('s1','s2','safety','monitor'):
        if role not in needed:continue
        c=configs[role]
        if c['batch_size']>len(members[role]):raise ValueError('Requested S1 batch exceeds available clients')
        workers.append(dict(worker_id='prototype-'+role,physical_group_id='prototype-gpu',component_id=role,role=role,
            model_revision=c['model_revision'],runtime_fingerprint=c['runtime_fingerprint'],hardware_kind=c['hardware_kind'],
            expected_memory_mib=1,profile_id='unprofiled-prototype',profile_source='unmeasured',batch_size=c['batch_size'],
            batch_wait_ms=c['batch_wait_ms'],resources=dict(num_cpus=1,num_gpus=int(mode=='gpu')),
            members=members[role],backend=deepcopy(c['binding'])))
        if queue_policy!='fifo':workers[-1]['group_scheduling']=dict(policy=queue_policy)
        if role in role_concurrency:workers[-1]['group_scheduling']['max_concurrent_invocations']=role_concurrency[role]
        if queue_policy=='edf_queue_expiry_v1':workers[-1]['group_scheduling']['queue_budget_ms']=c['deadline_ms']
    result=dict(schema_version='1.0',purpose=PURPOSE,mode=mode,classes=classes,workers=workers,
        inputs=dict(configs=deepcopy(configs),population=deepcopy(population),frequency_hz=frequency_hz,
                    action_time_s=action_time_s,s1_headroom_fraction=s1_headroom_fraction,queue_policy=queue_policy),
        source_manifest=runtime_source_manifest(),required_physical_GPUs=int(mode=='gpu'),
        paper_model_placement=not extension,single_gpu_colocation_extension=extension,
        invocation_policy={'fifo':'serialized FIFO across models','edf_queue_expiry_v1':'non-preemptive earliest worker deadline; expire queued requests','parallel_roles_v1':'overlapping distinct roles; serialized within each role; no GPU isolation guarantee','parallel_roles_batch_s1_v1':'overlapping distinct roles; bounded S1 batching; serialized within each role; no GPU isolation guarantee'}[queue_policy] if extension else 'S1 shared dynamic batching',
        physical_actor_resources=dict(num_cpus=8 if mode=='gpu' else 1,num_gpus=int(mode=='gpu')),
        formal_performance_claim=False,R07_profile_adoption=False,semantic_quality_verified=False,
        scope='bounded connection validation with raw SLO evaluation; no capacity or semantic-quality certification')
    if periodic_phases!='synchronized':result['inputs']['periodic_phases']=periodic_phases
    if collision_guard is not None:
        result['inputs']['collision_guard']=deepcopy(collision_guard);result['collision_guard']=deepcopy(collision_guard)
        result['safety_scope']='simulation-only hybrid: VLM people/visibility plus prospective MuJoCo geometry; separate from VLM-only baseline'
    if recovery_policy is not None:
        result['inputs']['recovery_policy']=deepcopy(recovery_policy);result['recovery_policy']=deepcopy(recovery_policy)
        result['recovery_scope']='separately audited bounded simulation retreat; never a model-generated S1 action'
    if role_concurrency:
        result['inputs']['role_concurrency']=deepcopy(role_concurrency)
        result['invocation_policy']='bounded async VLM concurrency with serialized S1 batching; no GPU isolation guarantee'
    if prefetch_resets:
        result['inputs']['prefetch_resets']=True;result['prefetch_resets']=True
    if recovery_policy is not None:result['recovery_feedback_roles']=['s2']
    result['plan_id']=digest(result);return result


def validate(plan):
    if plan!=prepare(**plan['inputs']):raise ValueError('Prototype plan/config/source changed')
    return plan


def check_inventory(plan,inventory):
    validate(plan)
    if set(inventory)!={w['worker_id'] for w in plan['workers']}:raise ValueError('Prototype worker inventory incomplete')
    instances=set();parents=set();leases=set()
    for w in plan['workers']:
        v=inventory[w['worker_id']]
        if (v['plan_id'],v['worker_spec_sha256'],v['runtime_source_sha256'])!=(plan['plan_id'],digest(w),runtime_source_manifest()['sha256']):
            raise ValueError('Prototype worker binding/source mismatch')
        if v['instance_id'] in instances:raise ValueError('Logical model instance alias')
        instances.add(v['instance_id']);parents.add((v['node_id'],v['pid']))
        if plan['single_gpu_colocation_extension'] and v.get('physical_group_id')!='prototype-gpu':raise ValueError('Colocation group mismatch')
        if plan['mode']=='gpu':
            if v['execution_kind']!='external_model':raise ValueError('Real GPU models required')
            check_gpu_device(w,v['gpu_ids'],v['gpu_properties'],cuda_available=True,visible_count=1)
            leases.add((v['node_id'],str(v['gpu_ids'][0])))
        elif v['gpu_ids'] or v['execution_kind']!='cpu_dummy':raise ValueError('CPU provenance mismatch')
    if len(parents)!=1 or len(leases)!=int(plan['mode']=='gpu'):raise ValueError('Expected one physical actor and GPU lease')
    return dict(workers=len(inventory),physical_actors=1,unique_gpu_leases=len(leases),formal_performance_claim=False)
