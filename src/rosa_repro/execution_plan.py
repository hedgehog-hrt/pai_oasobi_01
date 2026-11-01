"""Compile a validated R07 witness to explicit, replayable worker bindings.
No Ray import and no GPU side effects. The compiler can also run without OR-Tools.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import re
from pathlib import Path

from .heterogeneous import Fleet,validate_fleet_schedule
from .multicomponent import ROLES

DUMMY_FACTORY='rosa_repro.component_runtime:dummy_backend'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def global_robot(cid,rid):return json.dumps([cid,rid],separators=(',',':'),ensure_ascii=False)


def runtime_source_manifest():
    paths=sorted([*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.json')])
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(files=hashes,sha256=digest(hashes))


def validate_binding(b,mode):
    if set(b) not in ({'factory','kwargs'},{'factory','kwargs','runtime_env'}) or not isinstance(b['factory'],str) or b['factory'].count(':')!=1 or not isinstance(b['kwargs'],dict):
        raise ValueError('Invalid backend factory binding')
    if 'runtime_env' in b:
        env=b['runtime_env']
        if not isinstance(env,dict) or set(env) not in ({'py_executable'},{'py_executable','env_vars'}) or not isinstance(env['py_executable'],str) or not re.fullmatch(r'/[A-Za-z0-9_./-]+',env['py_executable']):
            raise ValueError('runtime_env requires a plain absolute py_executable path')
        if 'env_vars' in env:
            values=env['env_vars'];paths={'CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY'}
            if not isinstance(values,dict) or not paths<=set(values) or set(values)-paths-{'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_MPS_CLIENT_PRIORITY'}:
                raise ValueError('runtime_env permits only paired MPS paths and optional thread percentage')
            if any(not isinstance(values[k],str) or not re.fullmatch(r'/[A-Za-z0-9_./-]+',values[k]) or '..' in values[k].split('/') for k in paths):
                raise ValueError('Plain absolute MPS paths required')
            if 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in values and (not isinstance(values['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'],str) or not re.fullmatch(r'(?:[1-9]|[1-9][0-9]|100)',values['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])):
                raise ValueError('MPS thread percentage must be an integer string in 1..100')
            if 'CUDA_MPS_CLIENT_PRIORITY' in values and values['CUDA_MPS_CLIENT_PRIORITY'] not in ('0','1'):
                raise ValueError('MPS client priority must be 0 or 1')
    if (mode=='cpu_dummy')!=(b['factory']==DUMMY_FACTORY):raise ValueError('Dummy/model execution mode mismatch')


def compile_execution_plan(fleet_input,schedule,roles,*,mode='cpu_dummy',bindings=None):
    fleet=Fleet.from_dict(fleet_input)
    if schedule.get('status')!='feasible':raise ValueError('Deployment requires a feasible R07 witness')
    validate_fleet_schedule(fleet,schedule['frequencies_hz'],schedule['placements'],schedule['routes'])
    if mode not in ('cpu_dummy','gpu'):raise ValueError('Unknown execution mode')
    components={m['id']:m for c in fleet_input['classes'] for m in c['components']}
    if set(roles)!=set(components):raise ValueError('Explicit role map must cover every component')
    for mid,m in components.items():
        if roles[mid] not in ('s1','s2','safety','monitor') or (m['role']=='obligation')!=(roles[mid] in ('safety','monitor')):
            raise ValueError('Role mapping disagrees with workload')
        if m['role']!='obligation' and roles[mid]!=m['role']:raise ValueError('Goal role mismatch')
    bindings=deepcopy(bindings or {})
    if mode=='cpu_dummy' and not bindings:
        bindings={mid:dict(factory=DUMMY_FACTORY,kwargs={}) for mid in components}
    if set(bindings)!=set(components):raise ValueError('Backend binding required for every component')
    for b in bindings.values():validate_binding(b,mode)
    classes=[]
    for c in fleet_input['classes']:
        specs=[]
        for m in c['components']:
            s=dict(id=m['id'],role=roles[m['id']],deadline_ms=m['deadline_ms'])
            if m['role']=='obligation':
                ns=Decimal('1000000000')/Decimal(str(m['periodic_hz']))
                if ns!=int(ns):raise ValueError('Periodic rate must have an integer-nanosecond period')
                s.update(period_ns=int(ns),phase_ns=0)
            specs.append(s)
        matches=[p for p,rs in ROLES.items() if rs=={s['role'] for s in specs} and len(rs)==len(specs)]
        if len(matches)!=1:raise ValueError('Execution bridge supports exactly P1-P4 role sets')
        audit=dict(id=c['id'],pipeline=matches[0],weight=c['weight'],s2_horizon=c['s2_horizon'],components=specs)
        robots=[dict(robot_id=global_robot(c['id'],str(i)),local_robot_id=str(i),
                     routes=deepcopy(schedule['routes'][c['id']][str(i)])) for i in range(c['robots'])]
        classes.append(dict(audit=audit,frequency_hz=schedule['frequencies_hz'][c['id']],
                            action_time_s=c['action_time_s'],robots=robots))
    workers=[]
    for w in schedule['placements']:
        m=components[w['component_id']]
        if w['batch_wait_ms']>1000:raise ValueError('Runtime batch wait limited to one second')
        workers.append(dict(worker_id=w['worker_id'],component_id=m['id'],role=roles[m['id']],
            model_revision=m['model_revision'],runtime_fingerprint=m['runtime_fingerprint'],profile_id=w['profile_id'],
            profile_source=fleet.profiles[w['profile_id']].source,expected_memory_mib=w['memory_mib'],
            hardware_kind=fleet.hardware['kind'],
            batch_size=w['batch_size'],batch_wait_ms=w['batch_wait_ms'],
            resources=dict(num_cpus=1,num_gpus=1 if mode=='gpu' else 0),
            members=[global_robot(x['class_id'],x['robot_id']) for x in w['members']],backend=bindings[m['id']]))
    inputs=dict(fleet=deepcopy(fleet_input),schedule=deepcopy(schedule),roles=deepcopy(roles),mode=mode,bindings=bindings)
    result=dict(schema_version='1.0',mode=mode,inputs=inputs,classes=classes,workers=workers,
                deployment_scope='static routes; one replica per worker; no admission or failover',
                kernel_execution_verified=False)
    result['plan_id']=digest(result)
    return result


def validate_execution_plan(plan):
    if plan.get('purpose')=='single_gpu_prototype_validation':
        from .prototype_plan import validate
        return validate(plan)
    if plan.get('purpose')=='dedicated_GPU_execution':
        from .dedicated_execution import validate
        return validate(plan)
    if plan.get('purpose')=='global_population_replan':
        from .global_planner import validate_plan
        return validate_plan(plan)
    if plan.get('purpose')=='paper_fixed_allocation_comparison':
        from .paper_baselines import validate
        return validate(plan)
    x=plan['inputs']
    expected=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode=x['mode'],bindings=x['bindings'])
    if plan!=expected:raise ValueError('Execution plan changed or its digest/derived fields disagree')
    return plan


def check_gpu_device(worker,ids,properties,*,cuda_available,visible_count):
    if len(ids)!=1 or not cuda_available or visible_count!=1:raise ValueError('Expected one allocated and visible CUDA device')
    if properties['name']!=worker['hardware_kind'] or not math.isfinite(properties['memory_mib']) or properties['memory_mib']<worker['expected_memory_mib']:
        raise ValueError('Actual GPU kind/memory differs from profile requirements')


def check_runtime_inventory(plan,inventory):
    """Require every planned worker, distinct process identities and GPU leases."""
    if plan.get('purpose')=='single_gpu_prototype_validation':
        from .prototype_plan import check_inventory
        return check_inventory(plan,inventory)
    if plan.get('purpose')=='dedicated_GPU_execution':
        from .dedicated_execution import check_inventory
        return check_inventory(plan,inventory)
    if set(inventory)!={w['worker_id'] for w in plan['workers']}:raise ValueError('Worker inventory incomplete')
    instances=set();devices=set()
    for w in plan['workers']:
        r=inventory[w['worker_id']]
        if r['plan_id']!=plan['plan_id'] or r['worker_spec_sha256']!=digest(w):raise ValueError('Runtime binding mismatch')
        if r['runtime_source_sha256']!=runtime_source_manifest()['sha256']:raise ValueError('Worker source differs from driver')
        if r['instance_id'] in instances:raise ValueError('Worker handle alias detected')
        instances.add(r['instance_id'])
        ids=r['gpu_ids']
        if plan['mode']=='cpu_dummy':
            if ids or r['execution_kind']!='cpu_dummy':raise ValueError('CPU provenance mismatch')
        else:
            if len(ids)!=1 or r['execution_kind']!='external_model':raise ValueError('GPU lease/model provenance mismatch')
            check_gpu_device(w,ids,r['gpu_properties'],cuda_available=True,visible_count=1)
            device=(r['node_id'],str(ids[0]))
            if device in devices:raise ValueError('GPU assigned to multiple model workers')
            devices.add(device)
    return dict(workers=len(inventory),unique_gpu_leases=len(devices),gpu_kernel_execution_verified=False)
