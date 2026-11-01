"""R07 population replanning with stable robot IDs and bounded solver execution."""
import argparse,asyncio,json,math,os,sys
from copy import deepcopy
from pathlib import Path
from .execution_plan import compile_execution_plan,validate_execution_plan,digest,runtime_source_manifest,global_robot
from .heterogeneous import Fleet,HeterogeneousScheduler
from .regression import write_json,sha256

PURPOSE='global_population_replan'

def population(catalog,value):
    known={c['id'] for c in catalog['fleet']['classes']}
    if not isinstance(value,dict) or set(value)!=known:raise ValueError('Explicit population for every catalog class required')
    ids=[]
    for cid,robots in value.items():
        if not isinstance(robots,list) or len(robots)>64 or any(not isinstance(r,str) or not r or len(r)>128 for r in robots):raise ValueError('Invalid population/identity')
        ids+=robots
    if len(ids)!=len(set(ids)):raise ValueError('Robot identity reused across classes')
    return {cid:sorted(value[cid]) for cid in sorted(value)}


def catalog_from_plan(plan):
    validate_execution_plan(plan)
    if plan.get('purpose'):raise ValueError('Original capped R07 reference required')
    return {k:deepcopy(plan['inputs'][k]) for k in ('fleet','roles','bindings','mode')} | {'seed_frequencies_hz':deepcopy(plan['inputs']['schedule']['frequencies_hz'])}


def validate_catalog(c):
    if set(c)!={'fleet','roles','bindings','mode','seed_frequencies_hz'}:raise ValueError('Invalid global catalog')
    f=Fleet.from_dict(c['fleet']);comps={m['id'] for cl in c['fleet']['classes'] for m in cl['components']}
    if set(c['roles'])!=comps or set(c['bindings'])!=comps or set(c['seed_frequencies_hz'])!=set(f.classes):raise ValueError('Catalog model/frequency coverage differs')
    from .execution_plan import validate_binding
    if c['mode'] not in ('cpu_dummy','gpu'):raise ValueError('Invalid catalog mode')
    for b in c['bindings'].values():validate_binding(b,c['mode'])
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in c['seed_frequencies_hz'].values()):raise ValueError('Invalid seed rates')
    if not 0<c['fleet']['search']['time_limit_s']<=300:raise ValueError('Solver must have a 0..300s finite budget')
    return c


def remap(base,robots,revision):
    # The mapping is derived, not a free-form override of a feasible witness.
    validate_execution_plan(base)
    if base.get('purpose') or type(revision) is not int or revision<0:raise ValueError('Invalid base/revision')
    if set(robots)!={c['audit']['id'] for c in base['classes']}:raise ValueError('Mapping classes differ')
    result=deepcopy(base);aliases={};workers={w['worker_id']:f'g{revision}-{w["worker_id"]}' for w in base['workers']}
    for c in result['classes']:
        cid=c['audit']['id'];names=robots[cid]
        if len(names)!=len(c['robots']) or len(set(names))!=len(names):raise ValueError('Mapping population differs')
        for old,new in zip(c['robots'],names):
            aliases[old['robot_id']]=new;old['robot_id']=new;old['routes']={m:workers[w] for m,w in old['routes'].items()}
    if len(set(aliases.values()))!=len(aliases):raise ValueError('Cross-class alias')
    for w in result['workers']:w['worker_id']=workers[w['worker_id']];w['members']=[aliases[r] for r in w['members']]
    result.update(purpose=PURPOSE,population_inputs=dict(base=deepcopy(base),robots=deepcopy(robots),revision=revision),deployment_scope='global R07 plan with stable robot identity; migration managed separately')
    result.pop('plan_id');result['plan_id']=digest(result);return result


def validate_plan(plan):
    x=plan['population_inputs']
    if plan!=remap(x['base'],x['robots'],x['revision']):raise ValueError('Population plan changed')
    return plan


def solve(catalog,robots,gpu_budget,revision):
    validate_catalog(catalog);robots=population(catalog,robots)
    if type(gpu_budget) is not int or not 0<=gpu_budget<=128:raise ValueError('Explicit physical budget required')
    active={cid:rs for cid,rs in robots.items() if rs};base=dict(kind='global_plan_decision',population=robots,gpu_budget=gpu_budget,revision=revision,source_manifest=runtime_source_manifest())
    if not active:return dict(base,status='idle',plan=None,solver=None)
    f=deepcopy(catalog['fleet']);f['classes']=[dict(c,robots=len(active[c['id']])) for c in f['classes'] if c['id'] in active];f['hardware']['gpu_count']=gpu_budget
    ids={m['id'] for c in f['classes'] for m in c['components']};f['profiles']=[p for p in f['profiles'] if p['component_id'] in ids]
    scheduler=HeterogeneousScheduler(Fleet.from_dict(f));s=scheduler.solve();method='R07_adaptive_frontier'
    if s['status']!='feasible':
        # A known positive seed can still provide a witness when zero load is
        # outside measured profile support. Never invent a zero-rate sample.
        fallback=scheduler.check_rates({cid:catalog['seed_frequencies_hz'][cid] for cid in active})
        if fallback['status']=='feasible':s=fallback;method='validated_reference_rate_fallback'
    if s['status']!='feasible':return dict(base,status='unresolved',plan=None,solver=s)
    p=compile_execution_plan(f,s,{m:catalog['roles'][m] for m in ids},mode=catalog['mode'],bindings={m:catalog['bindings'][m] for m in ids})
    return dict(base,status='planned',plan=remap(p,active,revision),solver=s,method=method,optimality_claim=False)


class ProcessPlanner:
    def __init__(self,python):
        p=Path(python)
        if not p.is_absolute() or not p.is_file():raise ValueError('Explicit existing solver Python required')
        self.python=str(p);self.python_sha256=sha256(p.resolve())
    async def __call__(self,catalog,robots,budget,revision,directory):
        from .resume_workflow import supervise
        out=Path(directory);out.mkdir(parents=True,exist_ok=False)
        request=dict(catalog=catalog,robots=robots,gpu_budget=budget,revision=revision);write_json(out/'input.json',request)
        cmd=[self.python,'-m','rosa_repro.global_planner','--input',str((out/'input.json').resolve()),'--output',str((out/'decision.json').resolve())]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH',''))
        timeout=2*catalog['fleet']['search']['time_limit_s']+15
        write_json(out/'command.json',dict(argv=cmd,python_sha256=self.python_sha256,wall_budget_s=timeout))
        r=await asyncio.to_thread(supervise,cmd,env,out,timeout);write_json(out/'process.json',r)
        if r['exit_code'] or r['timed_out'] or r['interrupted']:return dict(status='unresolved',reason='solver_process_failed_or_timed_out',plan=None)
        value=json.loads((out/'decision.json').read_text())
        if value['source_manifest']!=runtime_source_manifest() or value['population']!=population(catalog,robots) or value['gpu_budget']!=budget or value['revision']!=revision:raise ValueError('Solver output identity differs')
        if value['plan']:validate_plan(value['plan'])
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_json(a.output,solve(**json.loads(a.input.read_text())))
