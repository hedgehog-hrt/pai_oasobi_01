"""Prospective finite changes and state-bank bindings for dynamic local/Pod trials."""
from copy import deepcopy
import math
from .global_control import validate as validate_spec
from .global_planner import population
from .execution_plan import digest,runtime_source_manifest
from .paired import load_state_bank
from .profile_adoption import inside
from .regression import sha256


def prepare(spec,banks,root,changes,*,warmup_s=1.,duration_s=120.,setup_budget_s=600.,max_steps=10000,max_requests=100000,max_capture_bytes=1024**3,health=None,evaluation=None,repetitions=1):
    validate_spec(spec);cat=spec['catalog'];classes={c['id']:c for c in cat['fleet']['classes']}
    if type(repetitions) is not int or not 1<=repetitions<=10 or (evaluation is not None and repetitions<3):raise ValueError('Evaluated dynamic scenarios require at least three repetitions')
    if evaluation is not None:
        from .dynamic_protocol import validate as validate_protocol
        validate_protocol(evaluation,spec,warmup_s,duration_s)
    if set(banks)!=set(classes):raise ValueError('Every catalog class requires an initial-state bank')
    if any(type(x) not in (float,int) or not math.isfinite(x) or not 0<x<=7200 for x in (warmup_s,duration_s,setup_budget_s)):raise ValueError('Finite timing budgets required')
    if type(max_steps) is not int or not 8<=max_steps<=1000000 or type(max_requests) is not int or not 1<=max_requests<=100000 or type(max_capture_bytes) is not int or not 1<=max_capture_bytes<=100*1024**3:raise ValueError('Bounded capture/request/step limits required')
    health=deepcopy(health or dict(interval_s=1.,timeout_s=5.,max_recoveries=3))
    if set(health)!={'interval_s','timeout_s','max_recoveries'} or not .05<=health['interval_s']<=60 or not .01<=health['timeout_s']<=60 or type(health['max_recoveries']) is not int or not 1<=health['max_recoveries']<=20:raise ValueError('Invalid health policy')
    if not isinstance(changes,list) or len(changes)>30:raise ValueError('At most thirty prospective changes')
    active=deepcopy(spec['initial_population']);robots={r:c for c,rs in active.items() for r in rs};previous=-1
    for event in changes:
        op=event.get('op');fields={'join':{'class_id','robot_id'},'leave':{'robot_id'},'budget':{'gpu_budget'},'replan':set(),'fault':{'robot_id','component_id'}}
        if op not in fields or set(event)!={'op','offset_s'}|fields[op]:raise ValueError('Unknown change fields')
        offset=event['offset_s']
        if type(offset) not in (float,int) or not math.isfinite(offset) or not 0<=offset<duration_s or offset<previous:raise ValueError('Changes must be ordered within fixed window')
        previous=offset
        if op=='join':
            cid=event['class_id'];rid=event['robot_id']
            if cid not in active or any(rid in rs for rs in active.values()) or (rid in robots and robots[rid]!=cid):raise ValueError('Invalid join/class reuse')
            active[cid].append(rid);robots[rid]=cid;population(cat,active)
        elif op=='leave':
            rid=event['robot_id']
            if not any(rid in rs for rs in active.values()):raise ValueError('Leave needs an active robot')
            active[robots[rid]].remove(rid)
        elif op=='budget':
            if type(event['gpu_budget']) is not int or not 0<=event['gpu_budget']<=128:raise ValueError('Invalid GPU budget change')
        elif op=='fault':
            rid=event['robot_id']
            if cat['mode']!='cpu_dummy':raise ValueError('Fault injection is CPU-only; GPU health recovery observes actual owned actor failures')
            if rid not in robots or rid not in active[robots[rid]] or event['component_id'] not in {m['id'] for m in classes[robots[rid]]['components']}:raise ValueError('Fault target not active')
    slots=sum(math.ceil((warmup_s+duration_s)*m['periodic_hz'])*sum(cid==c['id'] for cid in robots.values()) for c in classes.values() for m in c['components'] if m['role']=='obligation')
    if slots>90000:raise ValueError('Dynamic trial exceeds periodic audit budget')
    entries={}
    for cid,name in banks.items():
        path=inside(root,name);m,_=load_state_bank(path)
        if cat['mode']=='gpu' and m['source'].get('kind')!='LIBERO_benchmark_initial_states':raise ValueError('GPU requires actual LIBERO initial states')
        entries[cid]=dict(directory=name,manifest=m,manifest_sha256=sha256(path/'initial_states.json'),states_sha256=sha256(path/'states.npz'))
    value=dict(kind='prospective_global_scenario',spec=deepcopy(spec),banks=entries,changes=deepcopy(changes),robots=robots,
        evaluation=deepcopy(evaluation),repetitions=repetitions,
        warmup_s=warmup_s,duration_s=duration_s,setup_budget_s=setup_budget_s,max_steps=max_steps,max_requests=max_requests,max_capture_bytes=max_capture_bytes,
        health=health,source_manifest=runtime_source_manifest(),formal_GPU_performance_claim=False,
        wall_budget_s=math.ceil(setup_budget_s+warmup_s+duration_s+4*spec['deployment_timeout_s']+2*spec['call_timeout_s']+4*cat['fleet']['search']['time_limit_s']+90),
        semantics='one saved physical episode per robot across model redeployments; terminal outcomes remain terminal; no task-success inference from throughput')
    value['id']=digest(value);return value


def validate(value,root):
    if value!=prepare(value['spec'],{c:b['directory'] for c,b in value['banks'].items()},root,value['changes'],
        **{k:value[k] for k in ('warmup_s','duration_s','setup_budget_s','max_steps','max_requests','max_capture_bytes','health','evaluation','repetitions')}):raise ValueError('Scenario, source or state-bank identity changed')
    return value


def validate_evaluation(value):
    from .execution_plan import runtime_source_manifest
    base=deepcopy(value);base.pop('id')
    if digest(base)!=value['id'] or value['source_manifest']!=runtime_source_manifest():raise ValueError('Scenario/source digest changed')
    # Reporting checks the sealed prospective metadata; execution additionally
    # reconstructs this scenario against the actual state-bank files in validate.
    from .dynamic_protocol import validate as validate_protocol
    validate_spec(value['spec'])
    if value['evaluation'] is not None:
        validate_protocol(value['evaluation'],value['spec'],value['warmup_s'],value['duration_s'])
        if type(value['repetitions']) is not int or not 3<=value['repetitions']<=10:raise ValueError('Invalid prospective repetitions')
    return value
