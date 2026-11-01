"""Fixed-window, state-bank-bound live pipeline experiments (project convention)."""
from copy import deepcopy
from pathlib import Path
import math
from .execution_plan import validate_execution_plan,runtime_source_manifest,digest
from .paired import load_state_bank
from .regression import sha256
from .profile_adoption import inside


def nominal_rates(plan):
    result={}
    for c in plan['classes']:
        n=len(c['robots'])
        for m in c['audit']['components']:
            rate=(1e9/m['period_ns'] if m['role'] in ('safety','monitor') else
                  (1/c['action_time_s'] if c['frequency_hz'] is None else c['frequency_hz'])/c['audit']['s2_horizon'] if m['role']=='s2' else (1/c['action_time_s'] if c['frequency_hz'] is None else c['frequency_hz']))
            if n and rate>0:result[c['audit']['id']+'/'+m['id']]=n*rate
    return result


def protocol(plan,phase='formal',*,warmup_s=None,duration_s=None,request_timeout_s=45.,sim_timeout_s=60.,drain_timeout_s=60.,
             max_steps=720,execute_steps=8,max_episodes_per_robot=10000,max_requests=100000,max_capture_bytes=8*1024**3,warmup_min_success=None):
    validate_execution_plan(plan)
    if phase not in ('formal','cpu_smoke','gpu_connection'):raise ValueError('Unknown closed-loop protocol phase')
    prototype=plan.get('purpose')=='single_gpu_prototype_validation'
    visual_only=any(w.get('role')=='safety' and w.get('backend',{}).get('kwargs',{}).get('config',{}).get('structured_output')=='safety_visual_guard_v1' for w in plan['workers'])
    if visual_only and (not prototype or not plan.get('collision_guard')):raise ValueError('Visual-only Safety cannot authorize live actions without the geometric guard')
    if phase=='gpu_connection' and (not prototype or plan['mode']!='gpu'):
        raise ValueError('GPU connection protocol requires the explicit prototype plan')
    if prototype and phase=='formal':raise ValueError('Unprofiled prototype cannot claim a formal experiment')
    if (phase!='cpu_smoke')!=(plan['mode']=='gpu'):raise ValueError('GPU protocol requires GPU; CPU uses explicit smoke protocol')
    rates=nominal_rates(plan)
    if not rates:raise ValueError('No active component rates')
    low=min(rates.values());minimum=20 if phase=='formal' else 1 if phase=='gpu_connection' else 0;count=1000 if phase=='formal' else 1
    warm_count=minimum if warmup_min_success is None else warmup_min_success
    if type(warm_count) is not int or not minimum<=warm_count<=10000:raise ValueError('Invalid warmup minimum; formal requires at least 20')
    recommended=dict(warmup_s=max(30.,1.2*warm_count/low+request_timeout_s) if phase=='formal' else .05,
                     duration_s=max(120.,1.2*count/low+request_timeout_s) if phase=='formal' else .6)
    if phase=='gpu_connection':recommended=dict(warmup_s=max(10.,2/low),duration_s=max(20.,2/low))
    warmup_s=recommended['warmup_s'] if warmup_s is None else warmup_s
    duration_s=recommended['duration_s'] if duration_s is None else duration_s
    for v in (warmup_s,duration_s,request_timeout_s,sim_timeout_s,drain_timeout_s):
        if type(v) not in (float,int) or not math.isfinite(v) or v<=0:raise ValueError('Positive finite timing required')
    if warmup_s>3600 or duration_s>86400 or max(request_timeout_s,sim_timeout_s,drain_timeout_s)>3600:raise ValueError('Protocol timing cap exceeded')
    if phase=='formal' and (warmup_s<30 or duration_s<120):raise ValueError('Formal duration minima cannot be lowered')
    for v,upper in ((max_steps,10000),(execute_steps,40),(max_episodes_per_robot,100000),(max_requests,100000),(max_capture_bytes,100*1024**3)):
        if type(v) is not int or not 1<=v<=upper:raise ValueError('Invalid integer protocol cap')
    if execute_steps>max_steps:raise ValueError('Execution stride exceeds episode step budget')
    slots=sum(math.ceil((warmup_s+duration_s+drain_timeout_s)*1e9/m['period_ns'])*len(c['robots'])
        for c in plan['classes'] for m in c['audit']['components'] if m['role'] in ('safety','monitor'))
    if slots>100000:raise ValueError('Protocol exceeds evaluator periodic-slot cap')
    return dict(schema_version='2.0',phase=phase,warmup_s=float(warmup_s),duration_s=float(duration_s),request_timeout_s=float(request_timeout_s),
        sim_timeout_s=float(sim_timeout_s),drain_timeout_s=float(drain_timeout_s),max_steps=max_steps,execute_steps=execute_steps,
        max_episodes_per_robot=max_episodes_per_robot,max_requests=max_requests,max_capture_bytes=max_capture_bytes,
        min_warmup_success_per_active_class_component=warm_count,min_measure_success_per_active_class_component=count,
        warmup_admission='check each active class/component at fixed cutoff before new requests/actions; no automatic extension',
        recommended_timing=recommended,nominal_rates_hz=rates,nominal_rates_are_not_guaranteed=True,
        action_pacing='at_least_declared_chunk_wall_time_including_simulator_step',
        measurement_phase='connection_validation_closed_loop' if phase=='gpu_connection' else 'fixed_window_closed_loop',action_semantics='external_executor_reported_steps',
        capture_policy='every initial/post-step observation and every applied action; capture overhead remains in wall time',
        periodic_clock='global trial epoch; no catch-up burst; inactive/reset gaps retained in wall-time denominator',
        episode_order='case order rotated by robot index and repetition, cyclic; original case seed retained',
        task_success_denominator='all started episodes; warmup/start-boundary cohorts and censoring reported separately',
        formal_performance_claim=False,paper_protocol_equivalence_verified=False)


def validate_protocol(plan,p):
    keys=('warmup_s','duration_s','request_timeout_s','sim_timeout_s','drain_timeout_s','max_steps','execute_steps',
          'max_episodes_per_robot','max_requests','max_capture_bytes')
    if p!=protocol(plan,p['phase'],warmup_min_success=p['min_warmup_success_per_active_class_component'],**{k:p[k] for k in keys}):raise ValueError('Protocol changed')
    return p


def prepare_suite(plan,banks,root,p,*,repetitions=3,setup_budget_s=600):
    validate_execution_plan(plan);validate_protocol(plan,p)
    if type(repetitions) is not int or not 1<=repetitions<=10 or (p['phase']=='formal' and repetitions<3):raise ValueError('Formal suites require at least three repetitions')
    if type(setup_budget_s) not in (int,float) or not math.isfinite(setup_budget_s) or not 1<=setup_budget_s<=7200:raise ValueError('Invalid setup budget')
    classes={c['audit']['id'] for c in plan['classes'] if c['robots']}
    if set(banks)!=classes:raise ValueError('A state bank is required for each active class')
    entries={}
    for cid,name in banks.items():
        directory=inside(root,name);m,states=load_state_bank(directory)
        if plan.get('collision_guard') and m['task']!=plan['collision_guard']['task']:raise ValueError('Hybrid guard task and state bank differ')
        if plan['mode']=='gpu' and m['source'].get('kind')!='LIBERO_benchmark_initial_states':raise ValueError('GPU requires actual LIBERO state banks')
        if len({x['state_sha256'] for x in m['cases']})!=len(m['cases']):raise ValueError('Duplicate state in bank')
        entries[cid]=dict(directory=name,manifest_sha256=sha256(directory/'initial_states.json'),states_sha256=sha256(directory/'states.npz'),manifest=m)
    s=dict(schema_version='1.0',plan=deepcopy(plan),protocol=deepcopy(p),banks=entries,source_manifest=runtime_source_manifest(),
        repetitions=repetitions,setup_budget_s=setup_budget_s,
        trial_wall_budget_s=math.ceil(setup_budget_s+p['warmup_s']+p['duration_s']+p['sim_timeout_s']+2*p['drain_timeout_s']+30),
        cases=[dict(case_id=f'closed-loop-{i:03d}',rep=i) for i in range(repetitions)],
        scope='fixed-window simulation experiment; CPU smoke never establishes formal GPU results')
    s['suite_id']=digest(s);return s


def validate_suite(s,root,*,current_source=True):
    expected=prepare_suite(s['plan'],{cid:x['directory'] for cid,x in s['banks'].items()},root,s['protocol'],
                           repetitions=s['repetitions'],setup_budget_s=s['setup_budget_s'])
    if not current_source:
        expected['source_manifest']=s['source_manifest'];expected.pop('suite_id');expected['suite_id']=digest(expected)
    if expected!=s:raise ValueError('Closed-loop suite/source/state banks changed')
    return s


def sample_check(protocol,classes,records,start,end):
    counts={k:dict(warmup=0,measure=0) for k in protocol['nominal_rates_hz']}
    robots={r['robot_id']:c['audit']['id'] for c in classes for r in c['robots']}
    for r in records:
        if r['record_type']!='request' or r['status']!='ok' or not r['valid_output']:continue
        key=robots[r['robot_id']]+'/'+r['component_id']
        if key not in counts:continue
        if r['sent_ns']<start and r['finished_ns']<start:counts[key]['warmup']+=1
        if start<=r['sent_ns']<end and r['finished_ns']<=end:counts[key]['measure']+=1
    short=[key for key,v in counts.items() if v['warmup']<protocol['min_warmup_success_per_active_class_component'] or
           v['measure']<protocol['min_measure_success_per_active_class_component']]
    return dict(counts=counts,shortfall=short,requirements_met=not short,p99_minimum_samples=1000,
                formal_protocol_met=protocol['phase']=='formal' and not short)


def warmup_check(protocol,classes,records,cutoff):
    counts={k:0 for k in protocol['nominal_rates_hz']}
    robots={r['robot_id']:c['audit']['id'] for c in classes for r in c['robots']}
    for row in records:
        if row['record_type']=='request' and row['status']=='ok' and row['valid_output'] and row['sent_ns']<cutoff and row['finished_ns']<cutoff:
            key=robots[row['robot_id']]+'/'+row['component_id']
            if key in counts:counts[key]+=1
    minimum=protocol['min_warmup_success_per_active_class_component'];short=sorted(k for k,n in counts.items() if n<minimum)
    return dict(cutoff_ns=cutoff,counts=counts,required_per_active_class_component=minimum,shortfall=short,requirements_met=not short)
