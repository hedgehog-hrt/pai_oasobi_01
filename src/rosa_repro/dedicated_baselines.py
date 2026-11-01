"""Whole-workload resource preparation for dedicated-GPU paper baselines.
Multi-model co-residency is unresolved until joint memory/latency is measured.
"""
from copy import deepcopy
from .execution_plan import validate_execution_plan,digest,runtime_source_manifest
from .heterogeneous import Fleet


def prepare(reference,method,profiles,*,gpu_budget):
    validate_execution_plan(reference)
    if reference.get('purpose') or method not in ('gpu_per_robot','gpu_per_model_per_robot'):raise ValueError('Original ROSA reference and declared dedicated method required')
    if type(gpu_budget) is not int or not 0<=gpu_budget<=128:raise ValueError('Explicit physical GPU budget required')
    fleet=Fleet.from_dict(reference['inputs']['fleet']);required={m['id'] for c in reference['classes'] for m in c['audit']['components']}
    if set(profiles)!=required:raise ValueError('Explicit batch-one profile for every component required')
    chosen={}
    for mid,pid in profiles.items():
        p=fleet.profiles.get(pid)
        if not p or p.component_id!=mid or p.batch_size!=1:raise ValueError('Dedicated baseline requires matching B1 profiles')
        chosen[mid]=next(deepcopy(x) for x in reference['inputs']['fleet']['profiles'] if x['id']==pid)
    assignments=[];robots=[]
    for c in reference['classes']:
        for r in c['robots']:
            rid=r['robot_id'];robots.append(rid);models=[dict(component_id=m['id'],profile=chosen[m['id']],binding=deepcopy(reference['inputs']['bindings'][m['id']])) for m in c['audit']['components']]
            if method=='gpu_per_robot':assignments.append(dict(slot=len(assignments),robot_id=rid,models=models,reserved_physical_gpus=1))
            else:
                for model in models:assignments.append(dict(slot=len(assignments),robot_id=rid,models=[model],reserved_physical_gpus=1))
    cohabits=any(len(a['models'])>1 for a in assignments);needed=len(assignments);blockers=[]
    if needed>gpu_budget:blockers.append('insufficient_physical_GPU_budget_for_entire_workload')
    if cohabits:blockers.append('joint_model_memory_and_latency_measurement_required')
    # This artifact prepares resources only. Existing ComponentWorker allocates one
    # full GPU per model; it must never silently emulate co-residency with many GPUs.
    blockers.append('dedicated_executor_and_measured_runtime_validation_required')
    value=dict(kind='dedicated_GPU_resource_plan',method=method,reference_plan_id=reference['plan_id'],reference=deepcopy(reference),profiles=deepcopy(profiles),
        gpu_budget=gpu_budget,required_physical_GPUs=needed,shortfall=max(0,needed-gpu_budget),all_robots=robots,assignments=assignments,
        unused_GPU_budget=max(0,gpu_budget-needed),status='resource_budget_fits' if needed<=gpu_budget else 'insufficient_resources',
        executable=False,blocking_reasons=blockers,source_manifest=runtime_source_manifest(),frequency_cap_hz=None,
        model_memory_semantics='single-model profile declarations are not joint residency peaks; do not sum them as measured GPU use',
        robot_subset_dropped=False,paper_performance_claim=False)
    value['id']=digest(value);return value


def validate(value):
    if value!=prepare(value['reference'],value['method'],value['profiles'],gpu_budget=value['gpu_budget']):raise ValueError('Dedicated resource plan modified')
    return value
