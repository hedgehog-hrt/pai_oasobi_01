from copy import deepcopy
import pytest
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler,validate_fleet_schedule
from rosa_repro.heterogeneous_oracle import exhaustive_grid
from hetero_helpers import task,make_fleet,weighted_fixture


def test_weighted_frontier_minimums_and_independent_oracle():
    d=weighted_fixture();f=Fleet.from_dict(d);r=HeterogeneousScheduler(f).solve()
    assert r['status']=='feasible' and 6.98<=r['weighted_objective']<=7+1e-9
    assert r['frequencies_hz']['low']>=1 and r['frequencies_hz']['high']>1.99
    validate_fleet_schedule(f,r['frequencies_hz'],r['placements'],r['routes'])
    oracle=exhaustive_grid(d,{'high':[0,1,1.5,2,2.5,3,4],'low':[0,1,1.5,2,2.5,3,4]})
    assert oracle['status']=='grid_optimal' and oracle['best']['weighted_objective']==7
    assert not r['optimality_proven']


def test_switching_weights_changes_allocated_frequency():
    d=weighted_fixture();d['classes'][0]['weight']=1;d['classes'][1]['weight']=3
    r=HeterogeneousScheduler(Fleet.from_dict(d)).solve()
    assert r['frequencies_hz']['low']>1.99 and r['frequencies_hz']['high']>=1


def test_mix2_compacts_same_model_and_preserves_obligations():
    d=make_fleet([task('p2','P2',2),task('p4','P4',2)],gpus=4)
    f=Fleet.from_dict(d);r=HeterogeneousScheduler(f).check_rates({'p2':2.,'p4':2.})
    assert r['status']=='feasible' and len(r['placements'])==4
    assert r['isolated_goal_workers']==3 and r['compacted_goal_workers']==2
    assert r['goal_merges'] and r['obligations']['merges']
    assert {p['component_id'] for p in r['placements']}=={'s1','s2','safety','monitor'}
    validate_fleet_schedule(f,r['frequencies_hz'],r['placements'],r['routes'])


def test_mix4_has_complete_class_specific_routes():
    d=make_fleet([task(f'p{i}',f'P{i}') for i in range(1,5)],gpus=4)
    f=Fleet.from_dict(d);r=HeterogeneousScheduler(f).check_rates({f'p{i}':2 for i in range(1,5)})
    assert r['status']=='feasible' and len(r['placements'])==4
    assert [len(r['routes'][f'p{i}']['0']) for i in range(1,5)]==[1,2,3,4]
    oracle=exhaustive_grid(d,{f'p{i}':[2.] for i in range(1,5)})
    assert oracle['status']=='grid_optimal' and oracle['best']['weighted_objective']==8


def test_compaction_rechecks_strictest_class_slo():
    d=make_fleet([task('a'),task('b')],gpus=1)
    d['classes'][0]['components'][0]['deadline_ms']=150
    d['profiles'][0]['points']=[dict(rate_hz=0,mean_ms=50,p99_ms=100),dict(rate_hz=1,mean_ms=50,p99_ms=100),dict(rate_hz=20,mean_ms=100,p99_ms=200)]
    f=Fleet.from_dict(d);r=HeterogeneousScheduler(f).check_rates({'a':1.,'b':1.})
    assert r['status']=='unknown' and r['category']=='heuristic_rejection'


def test_different_components_cannot_share_gpu_even_same_model_revision():
    d=make_fleet([task('a','P4')],gpus=3)
    assert HeterogeneousScheduler(Fleet.from_dict(d)).solve()['status']=='infeasible'


def test_profile_identity_mismatch_rejected():
    d=weighted_fixture();d['classes'][1]['components'][0]['model_revision']='different'
    with pytest.raises(ValueError,match='identity'):Fleet.from_dict(d)


def test_missing_profiles_time_budget_and_nonmonotone_remain_unknown():
    d=weighted_fixture();d['profiles']=[]
    assert HeterogeneousScheduler(Fleet.from_dict(d)).solve()['status']=='unknown'
    d=weighted_fixture();d['search']['time_limit_s']=0
    assert HeterogeneousScheduler(Fleet.from_dict(d)).solve()['status']=='unknown'
    d=weighted_fixture();d['profiles'][0]['points'][1]['mean_ms']=1
    assert HeterogeneousScheduler(Fleet.from_dict(d)).solve()['status']=='unknown'


def test_fmin_physical_bound_infeasible_and_empty_class_supported():
    d=weighted_fixture();d['classes'][0]['min_action_rate_hz']=6
    assert HeterogeneousScheduler(Fleet.from_dict(d)).solve()['status']=='infeasible'
    d=make_fleet([task('empty',robots=0)],gpus=0)
    r=HeterogeneousScheduler(Fleet.from_dict(d)).solve()
    assert r['status']=='feasible' and r['weighted_objective']==0 and not r['placements']


def test_validator_rejects_forged_rate_route_duplicate_and_latency():
    f=Fleet.from_dict(weighted_fixture());base=HeterogeneousScheduler(f).solve()
    for corruption in ('rate','route','duplicate','latency'):
        r=deepcopy(base)
        if corruption=='rate':r['frequencies_hz']['high']=3
        elif corruption=='route':r['routes']['low']['0']['s1']='no-gpu'
        elif corruption=='duplicate':r['placements'][0]['members'].append(r['placements'][0]['members'][0])
        else:r['placements'][0]['mean_ms']=0
        with pytest.raises(ValueError):validate_fleet_schedule(f,r['frequencies_hz'],r['placements'],r['routes'])


def test_greedy_counterexample_is_unknown_not_infeasible():
    from hetero_helpers import greedy_counterexample
    d=greedy_counterexample();rates=dict(a=4.,b=3.,c=2.,d=2.)
    r=HeterogeneousScheduler(Fleet.from_dict(d)).check_rates(rates)
    assert r['status']=='unknown' and r['category']=='heuristic_rejection'
    assert r['compacted_goal_workers']==3
    oracle=exhaustive_grid(d,{c:[f] for c,f in rates.items()})
    assert oracle['status']=='grid_optimal' and oracle['best']['weighted_objective']==11
    assert sorted(w['rate_hz'] for w in oracle['best']['placements'])==[5,6]


def test_random_small_schedules_checked_by_independent_partition_oracle():
    import random
    rng=random.Random(20261028)
    for _ in range(32):
        classes=[task(f'c{i}',weight=rng.choice([0,1,2,3])) for i in range(rng.choice([2,3,4]))]
        d=make_fleet(classes,gpus=rng.choice([1,2,3]),capacity=rng.choice([2,3,4,6,8]))
        rates={c['id']:rng.choice([0.,1.,2.,3.,4.]) for c in classes}
        r=HeterogeneousScheduler(Fleet.from_dict(d)).check_rates(rates)
        oracle=exhaustive_grid(d,{c:[v] for c,v in rates.items()})
        assert oracle['status'] in ('grid_optimal','grid_infeasible')
        if r['status']=='feasible':
            assert oracle['status']=='grid_optimal'
            assert r['weighted_objective']==oracle['best']['weighted_objective']
        elif r['status']=='infeasible':assert oracle['status']=='grid_infeasible'
        else:assert r['status']=='unknown'


def test_shared_periodic_model_sums_distinct_class_rates():
    d=make_fleet([task('a','P2'),task('b','P2')],gpus=2)
    d['classes'][1]['components'][1]['periodic_hz']=1.
    r=HeterogeneousScheduler(Fleet.from_dict(d)).check_rates(dict(a=1.,b=2.))
    assert r['status']=='feasible'
    assert next(w for w in r['placements'] if w['component_id']=='monitor')['rate_hz']==1.5


def test_memory_limit_and_closed_loop_rechecked_after_merge():
    d=make_fleet([task('a'),task('b')],gpus=1)
    d['profiles'][0]['memory_mib']=25000
    assert HeterogeneousScheduler(Fleet.from_dict(d)).check_rates(dict(a=1.,b=1.))['status']=='unknown'
    d=make_fleet([task('a'),task('b')],gpus=1)
    d['profiles'][0]['points']=[dict(rate_hz=0,mean_ms=50,p99_ms=100),
        dict(rate_hz=4,mean_ms=50,p99_ms=100),dict(rate_hz=8,mean_ms=100,p99_ms=100),
        dict(rate_hz=20,mean_ms=100,p99_ms=100)]
    r=HeterogeneousScheduler(Fleet.from_dict(d)).check_rates(dict(a=4.,b=4.))
    assert r['status']=='unknown' and r['category']=='heuristic_rejection'
    assert exhaustive_grid(d,dict(a=[4.],b=[4.]))['status']=='grid_infeasible'


def test_oracle_unknown_missing_profiles_timeout_and_size_guard():
    d=weighted_fixture();d['profiles']=[]
    assert exhaustive_grid(d,dict(high=[1.],low=[1.]))['status']=='unknown'
    d=weighted_fixture()
    assert exhaustive_grid(d,dict(high=[1.],low=[1.]),time_limit_s=0)['status']=='unknown'
    d['classes'][0]['robots']=4
    with pytest.raises(ValueError,match='four robots'):exhaustive_grid(d,dict(high=[1.],low=[1.]))


def test_zero_weight_is_not_allocated_extra_rate_and_zero_missing_profile_unknown():
    d=weighted_fixture();d['classes'][1]['weight']=0
    r=HeterogeneousScheduler(Fleet.from_dict(d)).solve()
    assert r['status']=='feasible' and r['frequencies_hz']['low']==1.
    d=make_fleet([task('a'),task('b')],gpus=2)
    d['profiles'][0]['points'][0]['rate_hz']=.1
    assert HeterogeneousScheduler(Fleet.from_dict(d)).solve()['status']=='unknown'
