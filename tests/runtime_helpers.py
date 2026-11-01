from copy import deepcopy
from rosa_repro.execution_plan import compile_execution_plan
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler
from hetero_helpers import task,make_fleet


def plan_for(pipelines=('P4',),*,fast=True,bindings=None):
    classes=[task('c'+str(i),p) for i,p in enumerate(pipelines)]
    data=make_fleet(classes,gpus=len({m['id'] for c in classes for m in c['components']}),capacity=10000)
    if fast:
        for c in classes:
            c['action_time_s']=.005;c['s2_horizon']=3
            for m in c['components']:
                if m['id']=='safety':m.update(periodic_hz=50.,deadline_ms=20.)
                if m['id']=='monitor':m.update(periodic_hz=20.,deadline_ms=50.)
        for p in data['profiles']:
            p['capacity_bound_hz']=10000
            p['points']=[dict(rate_hz=0,mean_ms=1,p99_ms=2),dict(rate_hz=10000,mean_ms=1,p99_ms=2)]
    rates={c['id']:30. if fast else 2. for c in classes}
    r=HeterogeneousScheduler(Fleet.from_dict(data)).check_rates(rates)
    assert r['status']=='feasible'
    roles={m['id']:m['id'] for c in classes for m in c['components']}
    return compile_execution_plan(data,r,roles,bindings=bindings)
