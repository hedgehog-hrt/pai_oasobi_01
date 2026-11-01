"""Prospective paper comparisons; unresolved measurements are never executable plans."""
from copy import deepcopy
from fractions import Fraction
import random
from .execution_plan import digest

COMPONENTS = {'P1': ('s1',), 'P2': ('s1', 'monitor'),
              'P3': ('s1', 'safety', 'monitor'), 'P4': ('s1', 's2', 'safety', 'monitor')}
BATCHES = (1, 2, 4, 8, 16)


def allocation(components, gpu_count, method, sizes=None):
    components = tuple(components)
    if not components or len(set(components)) != len(components) or set(components)-set(COMPONENTS['P4']):
        raise ValueError('Invalid components')
    if type(gpu_count) is not int or gpu_count < 1 or method not in ('equal', 'weighted'):
        raise ValueError('Invalid allocation controls')
    if gpu_count < len(components):
        return dict(status='infeasible', reason='fewer physical GPUs than required components', counts=None)
    if set(components) == set(COMPONENTS['P4']) and gpu_count == 8:
        counts = dict(zip(COMPONENTS['P4'], (2, 2, 2, 2) if method == 'equal' else (1, 3, 1, 3)))
        return dict(status='allocated_unprofiled', counts=counts, rule='paper_Table_3_override')
    if method == 'weighted' and sizes is None and len(components) > 1:
        return dict(status='unresolved', counts=None, reason='component model sizes required')
    weights = {c: Fraction(1) if method == 'equal' or len(components) == 1 else Fraction(str(sizes[c])) for c in components}
    if any(w <= 0 for w in weights.values()): raise ValueError('Positive model sizes required')
    remaining = gpu_count-len(components)
    shares = {c: remaining*w/sum(weights.values()) for c, w in weights.items()}
    counts = {c: 1 + int(s) for c, s in shares.items()}
    for c in sorted(components, key=lambda c: (-(shares[c]-int(shares[c])), c))[:gpu_count-sum(counts.values())]:
        counts[c] += 1
    return dict(status='allocated_unprofiled', counts=counts, rule='project_minimum_one_then_largest_remainder_component_id_tie')


def workload(name, robots):
    names = ('P2', 'P4') if name == 'Mix2' else tuple(COMPONENTS) if name == 'Mix4' else (name,)
    if any(p not in COMPONENTS for p in names) or type(robots) is not int or robots < 1 or robots % len(names):
        raise ValueError('Invalid workload population')
    return [dict(pipeline=p, robots=robots//len(names), components=list(COMPONENTS[p]), weight=1,
                 minimum_frequency_hz=None, action_time_s=.2, s1_deadline_ms=200,
                 s2=dict(reuse_length=10, deadline_ms=2000) if p == 'P4' else None,
                 safety=dict(frequency_hz=2, deadline_ms=500) if 'safety' in COMPONENTS[p] else None,
                 monitor=dict(frequency_hz=.5, deadline_ms=2000) if 'monitor' in COMPONENTS[p] else None)
            for p in names]


def generate(*, seed=42, repetitions=3, gpu_count=8, sizes=None, r05_selection=None, dedicated_budgets=None):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(repetitions) is not int or not 3<=repetitions<=100 or seed+repetitions>=2**32:
        raise ValueError('Seed and at least three independent confirmation repetitions required')
    dedicated=deepcopy(dedicated_budgets or {})
    if set(dedicated)-{'gpu_per_robot','gpu_per_model_per_robot'} or any(type(n) is not int or not 0<=n<=128 for n in dedicated.values()):raise ValueError('Explicit dedicated GPU budgets from zero to 128 required')
    # Selections bind an externally predicted candidate; this generator does not infer one from pilot outcomes.
    groups = [(k, t, f'k{k}-a{int(t*1000)}') for k in (1, 4, 16) for t in (.05, .2, 1.)]
    selection = deepcopy(r05_selection or {})
    if set(selection)-{g for _, _, g in groups}: raise ValueError('Unknown R05 group')
    for s in selection.values():
        if set(s) != {'batch_size', 'prediction_evidence_sha256'} or type(s['batch_size']) is not int or s['batch_size'] not in BATCHES:
            raise ValueError('Invalid predicted selection')
        h=s['prediction_evidence_sha256']
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h): raise ValueError('Prediction evidence hash required')
    cells=[]
    def add(**cell):
        cell['id']=f"{cell['experiment'].lower()}-{len(cells):04d}";cells.append(cell)
    for k,t,g in groups:
        base=dict(experiment='R05',group=g,robots=k,action_time_s=t,latency_slo_ms=None,
                  slo_policy='none',physical_gpus=1,scope='one_GPU_adaptation; paper_Fig14_worker_count_unresolved',
                  frequency_cap_hz=None,phase_protocol='must_bind_measured_duration_and_sample_budget')
        for b in BATCHES:add(**base,stage='pilot',batch_size=b,repeat=0,aliases=[])
        for rep in range(repetitions):
            chosen=selection.get(g)
            add(**base,stage='confirmation',batch_size=1,repeat=rep,
                aliases=['predicted_best'] if chosen and chosen['batch_size']==1 else [],candidate='batch_one')
            if not chosen or chosen['batch_size']!=1:
                add(**base,stage='confirmation',batch_size=chosen['batch_size'] if chosen else None,
                    repeat=rep,aliases=[],candidate='predicted_best',prediction=chosen)
    for experiment, names, ks in [('R08', tuple(COMPONENTS), (8,16,32,64)), ('R09', ('Mix2','Mix4'), (8,16,32,64))]:
        for name in names:
            for k in ks:
                w=workload(name,k);components=tuple(c for c in COMPONENTS['P4'] if any(c in row['components'] for row in w))
                partitions={m:allocation(components,gpu_count,m,sizes) for m in ('equal','weighted')}
                for rep in range(repetitions):
                    base=dict(experiment=experiment,workload=name,classes=w,physical_gpus=gpu_count,repeat=rep,
                              profile_ids=None, runtime_id=None, input_corpus_id=None,
                              hardware_feasibility='unresolved_until_profile_and_runtime_validation')
                    for kind in ('full_baseline','placement_only'):
                        for method in ('equal','weighted'):
                            if name=='P1' and gpu_count==1 and method=='weighted':continue
                            add(**base,comparison=kind,method=method,allocation=partitions[method],
                                aliases=['weighted'] if name=='P1' and gpu_count==1 else [],
                                s1_batch_size=1 if kind=='full_baseline' else None,
                                s1_s2_rate_control='no_optimized_cap' if kind=='full_baseline' else 'identical_to_bound_ROSA_reference',
                                periodic_control='fixed_Table_1',
                                fixed_ROSA_batch_and_rates_sha256=None if kind=='placement_only' else 'not_applicable')
                    add(**base,comparison='ROSA',method='ROSA',allocation=None,aliases=[],
                        selected_execution_plan_sha256=None)
                    for method,budget in sorted(dedicated.items()):
                        needed=k if method=='gpu_per_robot' else sum(x['robots']*len(x['components']) for x in w)
                        add(**base,comparison='dedicated_baseline',method=method,aliases=[],s1_batch_size=1,
                            dedicated_gpu_budget=budget,required_physical_gpus=needed,
                            allocation=dict(status='allocated_unprofiled' if needed<=budget else 'infeasible',counts=None,
                                shortfall=max(0,needed-budget),robot_subset_dropped=False),
                            s1_s2_rate_control='no_optimized_cap',periodic_control='fixed_Table_1',
                            residency='joint_review_required' if method=='gpu_per_robot' and len(components)>1 else 'individual_model_profiles',
                            resource_comparison='explicit different physical budgets; no equal-budget claim')
    rng=random.Random(seed)
    pilot=[i for i,c in enumerate(cells) if c.get('stage')=='pilot'];later=[i for i,c in enumerate(cells) if c.get('stage')!='pilot']
    rng.shuffle(pilot);rng.shuffle(later);order=pilot+later
    for c in cells:
        c['trial_seed']=seed+c.get('repeat',0)
        if c.get('stage')=='confirmation':c['requires_completed_pilot_group']=c['group']
    result=dict(schema_version='1.0',kind='prospective_paper_conditions',executable=False,
        inputs=dict(seed=seed,repetitions=repetitions,gpu_count=gpu_count,sizes=sizes,r05_selection=selection,dedicated_budgets=dedicated),
        paper='arXiv:2607.01088v1; Tables 1-3 and Fig.14',cells=cells,
        randomized_order=[cells[i]['id'] for i in order],
        blockers=['bind actual input/runtime/profile evidence','resolve selected batch and placement controls',
                  'validate physical inventory and SLO feasibility','R05 requires a genuinely SLO-free executor'],
        claims=dict(GPU_execution=False,optimality=False,performance=False))
    result['sha256']=digest(result)
    return result


def validate(document):
    expected=generate(**document['inputs'])
    if document!=expected: raise ValueError('Paper condition matrix differs from its declared design')
    return expected
