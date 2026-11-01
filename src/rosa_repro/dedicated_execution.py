"""Dedicated baselines with explicit physical groups and uncapped batch-one clients."""
from copy import deepcopy
from .execution_plan import digest,runtime_source_manifest
from .dedicated_baselines import validate as validate_resources

PURPOSE='dedicated_GPU_execution'


def compile_plan(resources):
    validate_resources(resources)
    if resources['shortfall']:raise ValueError('Entire workload exceeds dedicated physical GPU budget')
    ref=resources['reference'];fleet=deepcopy(ref['inputs']['fleet']);roles=ref['inputs']['roles']
    components={m['id']:m for c in fleet['classes'] for m in c['components']};workers=[];groups=[];routes={}
    for a in resources['assignments']:
        gid='dedicated-group-'+str(a['slot']);members=[];envs=[]
        for model in a['models']:
            mid=model['component_id'];p=model['profile'];m=components[mid];wid='dedicated-worker-'+str(len(workers))
            if p['batch_wait_ms']!=0:raise ValueError('Dedicated batch one requires zero batching wait')
            if p['memory_mib']>fleet['hardware']['memory_mib']:raise ValueError('Individual model exceeds physical memory')
            w=dict(worker_id=wid,physical_group_id=gid,component_id=mid,role=roles[mid],model_revision=m['model_revision'],
                runtime_fingerprint=m['runtime_fingerprint'],profile_id=p['id'],profile_source=p['source'],expected_memory_mib=p['memory_mib'],
                hardware_kind=fleet['hardware']['kind'],batch_size=1,batch_wait_ms=0,
                resources=dict(num_cpus=1,num_gpus=int(ref['mode']=='gpu')),members=[a['robot_id']],backend=deepcopy(model['binding']))
            workers.append(w);members.append(wid);routes[a['robot_id'],mid]=wid;envs.append(w['backend'].get('runtime_env'))
        if any(env!=envs[0] for env in envs):raise ValueError('Co-resident models require the same Python runtime environment')
        reservations=[model['binding']['kwargs'].get('config',{}).get('gpu_memory_utilization',0) for model in a['models']
            if model['binding']['kwargs'].get('config',{}).get('engine')=='vllm']
        if len(members)>1 and sum(reservations)>=1:raise ValueError('Co-resident vLLM reservations leave no GPU headroom; adopt explicitly budgeted configurations first')
        groups.append(dict(group_id=gid,worker_ids=members,resources=dict(num_cpus=1,num_gpus=int(ref['mode']=='gpu')),runtime_env=envs[0]))
    classes=deepcopy(ref['classes'])
    for c in classes:
        c['frequency_hz']=None
        for r in c['robots']:r['routes']={m:routes[r['robot_id'],m] for m in r['routes']}
    value=dict(schema_version='1.0',purpose=PURPOSE,mode=ref['mode'],classes=classes,workers=workers,physical_groups=groups,
        inputs=dict(resource_plan=deepcopy(resources),fleet=fleet),method=resources['method'],
        required_physical_GPUs=len(groups),joint_residency_review_required=any(len(g['worker_ids'])>1 for g in groups),
        source_manifest=runtime_source_manifest(),rate_control='uncapped',model_invocation_policy='serialized FIFO within each physical group',
        slo_feasibility_claim=False,predicted_objective=None,kernel_execution_verified=False,
        memory_semantics='process-wide resident GPU memory; per-model profiles do not establish joint fit')
    value['plan_id']=digest(value);return value


def validate(plan):
    if plan!=compile_plan(plan['inputs']['resource_plan']):raise ValueError('Dedicated execution plan changed')
    return plan


def check_inventory(plan,inventory):
    from .execution_plan import check_gpu_device
    validate(plan)
    if set(inventory)!={w['worker_id'] for w in plan['workers']}:raise ValueError('Dedicated inventory incomplete')
    workers={w['worker_id']:w for w in plan['workers']};instances=set();leases=set();processes=set()
    for g in plan['physical_groups']:
        process=None;lease=None
        for wid in g['worker_ids']:
            w=workers[wid];v=inventory[wid]
            if (v['plan_id'],v['worker_spec_sha256'],v['runtime_source_sha256'])!=(plan['plan_id'],digest(w),runtime_source_manifest()['sha256']):raise ValueError('Dedicated worker binding/source differs')
            if v.get('physical_group_id')!=g['group_id'] or v['instance_id'] in instances:raise ValueError('Dedicated group/instance alias')
            instances.add(v['instance_id']);p=(v['node_id'],v['pid'])
            if process is not None and process!=p:raise ValueError('Co-resident models run in different processes')
            process=p
            if plan['mode']=='cpu_dummy':
                if v['gpu_ids'] or v['execution_kind']!='cpu_dummy':raise ValueError('Dedicated CPU provenance differs')
            else:
                if v['execution_kind']!='external_model':raise ValueError('GPU model provenance differs')
                check_gpu_device(w,v['gpu_ids'],v['gpu_properties'],cuda_available=True,visible_count=1)
                current=(v['node_id'],str(v['gpu_ids'][0]))
                if lease is not None and lease!=current:raise ValueError('Co-resident models use different GPUs')
                lease=current
        if process in processes:raise ValueError('Physical groups share one actor process')
        processes.add(process)
        if lease is not None:
            if lease in leases:raise ValueError('Dedicated physical groups share one GPU')
            leases.add(lease)
    return dict(workers=len(inventory),physical_groups=len(processes),unique_gpu_leases=len(leases),gpu_kernel_execution_verified=False)


def launch_gate(plan,*,calibration=False,joint_review=None):
    validate(plan)
    if type(calibration) is not bool:raise ValueError('Explicit calibration mode required')
    if plan['mode']=='cpu_dummy':
        if joint_review is not None:raise ValueError('CPU evidence cannot adopt GPU residency')
        return dict(mode='cpu_dummy',comparison_eligible=False)
    if calibration:return dict(mode='GPU_joint_residency_calibration',comparison_eligible=False)
    if plan['joint_residency_review_required']:
        from .dedicated_residency import validate_review
        candidate=validate_review(plan,joint_review)
        return dict(mode='GPU_dedicated',comparison_eligible=True,joint_review=deepcopy(joint_review),candidate_id=candidate['id'])
    return dict(mode='GPU_dedicated',comparison_eligible=True)
