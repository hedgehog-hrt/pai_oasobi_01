"""Fixed Equal/Weighted allocations: structural validity without SLO feasibility.

Baseline overload is an experimental outcome. Never call it a feasible R07
schedule or replace an uncapped client with an arbitrary high frequency.
"""
from collections import Counter,defaultdict
from copy import deepcopy
from .execution_plan import digest,validate_execution_plan
from .paper_conditions import generate,validate as validate_paper

PURPOSE='paper_fixed_allocation_comparison'


def _workload(reference,cell):
    actual=Counter();expected={x['pipeline']:x for x in cell['classes']}
    if reference['inputs']['fleet']['hardware']['gpu_count']!=cell['physical_gpus']:raise ValueError('Reference GPU budget differs from paper cell')
    for c in reference['classes']:
        pipeline=c['audit']['pipeline'];actual[pipeline]+=len(c['robots']);want=expected.get(pipeline)
        if not want or c['action_time_s']!=want['action_time_s'] or c['audit']['weight']!=want['weight'] or c['audit']['s2_horizon']!=10:raise ValueError('Paper workload/action/weight/horizon differs')
        for m in c['audit']['components']:
            role=m['role'];deadline={'s1':200,'s2':2000,'safety':500,'monitor':2000}[role]
            if m['id']!=role or m['deadline_ms']!=deadline:raise ValueError('Paper model roles/deadlines differ')
            if role in ('safety','monitor') and (m['period_ns'],m['phase_ns'])!=((500000000 if role=='safety' else 2000000000),0):raise ValueError('Paper periodic clock differs')
    if actual!=Counter({x['pipeline']:x['robots'] for x in cell['classes']}):raise ValueError('Paper class population differs')
    if any(c['min_action_rate_hz']!=0 for c in reference['inputs']['fleet']['classes']):raise ValueError('Paper minimum-rate convention must be zero')


def compile_baseline(reference,paper,cell_id,baseline_profiles=None):
    validate_execution_plan(reference);validate_paper(paper)
    if reference.get('purpose')==PURPOSE:raise ValueError('Reference must be a validated ROSA schedule')
    matches=[c for c in paper['cells'] if c['id']==cell_id]
    if len(matches)!=1:raise ValueError('Unknown paper cell')
    cell=matches[0]
    if cell.get('comparison')=='dedicated_baseline':
        from .dedicated_comparison import compile_cell
        return compile_cell(reference,paper,cell_id,baseline_profiles)
    if cell['experiment'] not in ('R08','R09') or cell['comparison'] not in ('full_baseline','placement_only'):raise ValueError('Expected R08/R09 baseline cell')
    if cell['allocation']['status']!='allocated_unprofiled':raise ValueError('Unresolved/infeasible physical allocation')
    _workload(reference,cell);full=cell['comparison']=='full_baseline';chosen=deepcopy(baseline_profiles or {})
    fleet=reference['inputs']['fleet'];profiles={p['id']:p for p in fleet['profiles']};refworkers={w['worker_id']:w for w in reference['workers']}
    roles=reference['inputs']['roles'];components={m['id']:m for c in fleet['classes'] for m in c['components']}
    if full and set(chosen)!=set(components):raise ValueError('Full baseline requires explicit batch-one profile for every component')
    if not full and chosen:raise ValueError('Placement comparison inherits every routed batch configuration')
    classes=deepcopy(reference['classes']);buckets=defaultdict(list);routes={}
    for c in classes:
        if full:c['frequency_hz']=None
        for robot in c['robots']:
            for mid,wid in robot['routes'].items():
                pid=chosen[mid] if full else refworkers[wid]['profile_id'];p=profiles.get(pid)
                if not p or p['component_id']!=mid:raise ValueError('Baseline model/profile differs')
                if full and p['batch_size']!=1:raise ValueError('Full baseline must use batch one')
                if roles[mid]!='s1' and (p['batch_size']!=1 or p['batch_wait_ms']!=0):raise ValueError('VLM batching policy differs')
                if p['memory_mib']>fleet['hardware']['memory_mib']:raise ValueError('Baseline profile exceeds physical memory')
                buckets[mid,pid].append(robot['robot_id'])
    workers=[]
    for mid,total in cell['allocation']['counts'].items():
        keys=sorted(k for k in buckets if k[0]==mid)
        if total<len(keys):raise ValueError('Fixed allocation cannot preserve all reference batch configurations')
        counts={k:1 for k in keys};remaining=total-len(keys);weight=sum(len(buckets[k]) for k in keys)
        if not keys:raise ValueError('Allocated component has no robot stream')
        shares={k:remaining*len(buckets[k])/weight for k in keys}
        for k in keys:counts[k]+=int(shares[k])
        for k in sorted(keys,key=lambda k:(-(shares[k]-int(shares[k])),k))[:total-sum(counts.values())]:counts[k]+=1
        for key in keys:
            p=profiles[key[1]];m=components[mid]
            for index in range(counts[key]):
                members=buckets[key][index::counts[key]];wid='baseline-'+str(len(workers))
                workers.append(dict(worker_id=wid,component_id=mid,role=roles[mid],model_revision=m['model_revision'],
                    runtime_fingerprint=m['runtime_fingerprint'],profile_id=p['id'],profile_source=p['source'],expected_memory_mib=p['memory_mib'],
                    hardware_kind=fleet['hardware']['kind'],batch_size=p['batch_size'],batch_wait_ms=p['batch_wait_ms'],
                    resources=dict(num_cpus=1,num_gpus=int(reference['mode']=='gpu')),members=members,backend=deepcopy(reference['inputs']['bindings'][mid])))
                for rid in members:routes[rid,mid]=wid
    for c in classes:
        for robot in c['robots']:robot['routes']={mid:routes[robot['robot_id'],mid] for mid in robot['routes']}
    value=dict(schema_version='1.0',purpose=PURPOSE,mode=reference['mode'],classes=classes,workers=workers,
        inputs=dict(reference=deepcopy(reference),paper_inputs=deepcopy(paper['inputs']),cell_id=cell_id,baseline_profiles=chosen,fleet=deepcopy(fleet)),
        comparison=dict(cell=deepcopy(cell),reference_plan_id=reference['plan_id'],rate_control='uncapped' if full else 'reference_fixed',
            physical_allocation=deepcopy(cell['allocation']),routing='stable per-profile round robin; profile-group replicas proportional to stream count',
            reference_batch_and_rates_sha256=digest(dict(classes=reference['classes'],workers=reference['workers']))),
        slo_feasibility_claim=False,predicted_objective=None,deployment_scope='fixed physical allocation; measured overload is retained',kernel_execution_verified=False)
    value['plan_id']=digest(value);return value


def validate(plan):
    x=plan['inputs'];expected=compile_baseline(x['reference'],generate(**x['paper_inputs']),x['cell_id'],x['baseline_profiles'])
    if plan!=expected:raise ValueError('Baseline plan/binding changed')
    return plan
