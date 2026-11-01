from copy import deepcopy
from helpers import fixture,profile,component


def task(cid,pipeline='P1',robots=1,weight=1,minimum=0):
    name={'P1':'s1','P2':'p2','P3':'p4','P4':'p4'}[pipeline]
    d=fixture(name,robots=robots)
    comps=d['workload']['components']
    if pipeline=='P3':comps=[c for c in comps if c['role']!='s2']
    return dict(id=cid,robots=robots,weight=weight,min_action_rate_hz=minimum,
                action_time_s=.2,s2_horizon=10,components=comps)


def make_fleet(classes,gpus=4,capacity=20):
    ids={m['id'] for c in classes for m in c['components']}
    profiles=[profile('s1',mean=50,p99=100,capacity=capacity),
              profile('s2',mean=200,p99=800,capacity=10),
              profile('safety',mean=50,p99=200,capacity=20),
              profile('monitor',mean=50,p99=500,capacity=20)]
    return dict(schema_version='1.0',id='synthetic-heterogeneous',classes=classes,
        hardware=dict(kind='synthetic-gpu',gpu_count=gpus,memory_mib=24000),
        profiles=[p for p in profiles if p['component_id'] in ids],
        search=dict(objective_tolerance=.01,time_limit_s=20,max_iterations=256))


def weighted_fixture():return make_fleet([task('high',weight=3,minimum=1),task('low',weight=1,minimum=1)],gpus=1,capacity=3)


def greedy_counterexample():
    # Greedy merges 2+2 first; a feasible packing instead uses 4+2 and 3+2.
    return make_fleet([task(cid) for cid in ('a','b','c','d')],gpus=2,capacity=6)
