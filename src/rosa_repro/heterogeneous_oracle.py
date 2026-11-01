"""Small independent oracle: enumerate mixed-class stream set partitions.
Does not call the production scheduler, compactor, checker, or Profile.at.
Optimality, when reported, is ONLY over the explicit finite rate grid.
"""
import itertools
import math
import time

from .contracts import validate
from .heterogeneous import HETERO_SCHEMA


def partitions(items):
    if not items:
        yield [];return
    first,*rest=items
    for partition in partitions(rest):
        yield [[first]]+[list(g) for g in partition]
        for i in range(len(partition)):
            groups=[list(g) for g in partition];groups[i]=[first]+groups[i]
            yield groups


def raw_estimate(profile,rate):
    if rate>profile.get('capacity_bound_hz',math.inf):return None,False
    points=profile['points']
    if rate<points[0]['rate_hz'] or rate>points[-1]['rate_hz']:return None,True
    for p in points:
        if rate==p['rate_hz']:return (p['mean_ms'],p['p99_ms']),False
    for a,b in zip(points,points[1:]):
        if a['rate_hz']<rate<b['rate_hz']:
            mean=a['mean_ms']+(b['mean_ms']-a['mean_ms'])*(rate-a['rate_hz'])/(b['rate_hz']-a['rate_hz'])
            return (mean,max(a['p99_ms'],b['p99_ms'])),False
    raise AssertionError('Unreachable interpolation bracket')


def exhaustive_grid(data,frequency_grid,*,time_limit_s=15):
    validate(data,HETERO_SCHEMA)
    classes={c['id']:c for c in data['classes']};gpus=data['hardware']['gpu_count']
    if sum(c['robots'] for c in classes.values())>4 or gpus>6:
        raise ValueError('Independent oracle limited to four robots and six GPUs')
    if set(frequency_grid)!=set(classes):raise ValueError('Grid class mismatch')
    if not math.isfinite(time_limit_s) or time_limit_s<0:raise ValueError('Invalid oracle time budget')
    if any(not values or any(not math.isfinite(f) or f<0 for f in values) for values in frequency_grid.values()):
        raise ValueError('Invalid explicit frequency grid')
    count=math.prod(len(v) for v in frequency_grid.values())
    if count>100000:raise ValueError('Oracle grid exceeds 100000 vectors')
    components={cid:{m['id']:m for m in c['components']} for cid,c in classes.items()}
    stream_sets={}
    for cid,c in classes.items():
        for model in c['components']:
            stream_sets.setdefault(model['id'],[]).extend((cid,str(i)) for i in range(c['robots']))
    stream_sets={m:s for m,s in stream_sets.items() if s}
    ordered=sorted(classes)
    vectors=[]
    for vector in itertools.product(*(frequency_grid[c] for c in ordered)):
        rates=dict(zip(ordered,vector))
        if any((classes[c]['robots']==0 and rates[c]!=0) or rates[c]>1/classes[c]['action_time_s'] or
               rates[c]*classes[c]['robots']+1e-12<classes[c]['min_action_rate_hz'] for c in ordered):continue
        objective=sum(classes[c]['weight']*classes[c]['robots']*rates[c] for c in ordered)
        vectors.append((objective,rates))
    vectors.sort(key=lambda x:-x[0])
    started=time.monotonic();tested=0;best=None;unknown_upper=-math.inf;timed_out=False
    def check_time():
        if time.monotonic()-started>=time_limit_s:raise TimeoutError
    for objective,rates in vectors:
        if best is not None and objective<=best['weighted_objective']+1e-12:break
        unknown=False;options={}
        try:
            check_time()
            for model,streams in stream_sets.items():
                profiles=[p for p in data['profiles'] if p['component_id']==model]
                if not profiles:unknown=True
                all_packings=[]
                for partition in partitions(streams):
                    check_time()
                    if len(partition)>gpus:continue
                    block_options=[]
                    for group in partition:
                        rate=sum(components[c][model]['periodic_hz'] if components[c][model]['role']=='obligation'
                            else rates[c]/(classes[c]['s2_horizon'] if components[c][model]['role']=='s2' else 1) for c,r in group)
                        deadline=min(components[c][model]['deadline_ms'] for c,r in group)
                        choices=[]
                        for p in profiles:
                            if p['memory_mib']>data['hardware']['memory_mib']:continue
                            est,unresolved=raw_estimate(p,rate);unknown|=unresolved
                            if est is not None and est[1]<=deadline:
                                choices.append(dict(component_id=model,profile_id=p['id'],members=list(group),
                                                    mean_ms=est[0],p99_ms=est[1],rate_hz=rate))
                        block_options.append(choices)
                    for packing in itertools.product(*block_options):
                        check_time();all_packings.append(list(packing))
                options[model]=all_packings
            witness=None
            models=sorted(options)
            def combine(index,selected):
                nonlocal witness
                check_time()
                if len(selected)>gpus:return False
                if index==len(models):
                    latency={}
                    for w in selected:
                        for cid,rid in w['members']:latency[cid,rid,w['component_id']]=w['mean_ms']
                    for cid,c in classes.items():
                        for robot in range(c['robots']):
                            loop=c['action_time_s']
                            for model in c['components']:
                                if model['role']=='obligation':continue
                                loop+=latency[cid,str(robot),model['id']]/1000/(c['s2_horizon'] if model['role']=='s2' else 1)
                            if rates[cid]*loop>1+1e-12:return False
                    witness=selected;return True
                for packing in options[models[index]]:
                    if combine(index+1,selected+packing):return True
                return False
            feasible=combine(0,[])
            tested+=1
            if feasible:best=dict(frequencies_hz=rates,weighted_objective=objective,placements=witness)
            elif unknown:unknown_upper=max(unknown_upper,objective)
        except TimeoutError:
            timed_out=True;unknown_upper=max(unknown_upper,objective);break
    status=('grid_optimal' if best is not None and not timed_out and unknown_upper<=best['weighted_objective'] else
            'grid_infeasible' if best is None and not timed_out and unknown_upper==-math.inf else 'unknown')
    return dict(status=status,search_space='explicit_frequency_grid_exhaustive_mixed_stream_partitions',
        tested_vectors=tested,total_valid_grid_vectors=len(vectors),time_budget_reached=timed_out,
        best=best,unknown_objective_upper=None if unknown_upper==-math.inf else unknown_upper,
        continuous_optimality_claimed=False,gpu_execution=False)
