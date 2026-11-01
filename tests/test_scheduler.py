import copy
import math
import random

import pytest
from jsonschema import ValidationError

from rosa_repro.domain import Problem
from rosa_repro.scheduler import HomogeneousScheduler, validate_schedule
from rosa_repro import oracle
from helpers import fixture, profile, unresolved_packing


@pytest.mark.parametrize('name,expected', [
    ('s1', 2.5), ('batch_tradeoff', 1/.28), ('s1_s2', 1/.30), ('p2', 2.5), ('p4', 1/.30),
])
def test_analytical_limits_and_independent_grid(name, expected):
    p = Problem.from_dict(fixture(name))
    result = HomogeneousScheduler(p).solve()
    assert result['status'] == 'feasible'
    assert result['search_status'] == 'converged'
    assert expected-.01 <= result['frequency_hz'] <= expected+1e-12
    assert abs(result['frequency_hz']-oracle.max_frequency_grid(p)) <= .01
    validate_schedule(p, result['frequency_hz'], result['placements'], result['routes'])


def test_small_fleets_gpu_budgets_against_oracle():
    rng = random.Random(42)
    for _ in range(24):
        d = fixture(robots=rng.randint(1,4), gpus=rng.randint(1,3))
        d['profiles'] = [profile(mean=rng.choice([20,50,100]), p99=150,
                                 capacity=rng.choice([4,8,12,20]))]
        p = Problem.from_dict(d)
        result = HomogeneousScheduler(p).solve()
        reference = oracle.max_frequency_grid(p, .005)
        assert result['status'] == 'feasible'
        assert abs(result['frequency_hz']-reference) <= .01
        validate_schedule(p, result['frequency_hz'], result['placements'], result['routes'])


def test_exact_robot_coverage_not_capacity_rounding():
    d = fixture(robots=3, gpus=2)
    d['profiles'] = [profile(capacity=2)]
    p = Problem.from_dict(d)
    r = HomogeneousScheduler(p).check_frequency(1)
    assert r['status'] == 'feasible'
    assert sorted(x['streams'] for x in r['placements']) == [1,2]
    assert set(r['routes']) == {'0','1','2'}


def test_obligation_capacity_reserved_first():
    d = fixture('p2', gpus=3)
    d['profiles'][1] = profile('monitor', mean=100, p99=500, capacity=1)
    p = Problem.from_dict(d)
    r = HomogeneousScheduler(p).solve()
    assert r['status'] == 'feasible'
    assert sum(x['component_id']=='monitor' for x in r['placements']) == 2
    assert r['frequency_hz'] <= 2.5
    d['hardware']['gpu_count'] = 2
    assert HomogeneousScheduler(Problem.from_dict(d)).check_frequency(1)['status'] == 'infeasible'


@pytest.mark.parametrize('name,gpus', [('p2',1), ('p4',3), ('s1',0)])
def test_insufficient_physical_workers(name,gpus):
    assert HomogeneousScheduler(Problem.from_dict(fixture(name,gpus=gpus))).solve()['status'] == 'infeasible'


@pytest.mark.parametrize('change', ['memory','p99'])
def test_memory_and_tail_latency_filter(change):
    d = fixture()
    if change == 'memory':
        d['hardware']['memory_mib'] = 999
    else:
        d['workload']['components'][0]['deadline_ms'] = 99
    r = HomogeneousScheduler(Problem.from_dict(d)).check_frequency(1)
    assert r['status'] == 'infeasible'


def test_mean_not_p99_in_loop_equation():
    d = fixture(robots=1)
    d['profiles'] = [profile(mean=50, p99=199)]
    p = Problem.from_dict(d)
    assert HomogeneousScheduler(p).check_frequency(4)['status'] == 'feasible'
    assert HomogeneousScheduler(p).check_frequency(4.001)['status'] == 'infeasible'


def test_s2_amortization():
    d = fixture('s1_s2', robots=1)
    d['workload']['s2_horizon'] = 1
    r = HomogeneousScheduler(Problem.from_dict(d)).solve()
    assert 1/.75-.01 <= r['frequency_hz'] <= 1/.75


def test_missing_profiles_are_unknown():
    d = fixture()
    d['profiles'] = []
    assert HomogeneousScheduler(Problem.from_dict(d)).check_frequency(1)['status'] == 'unknown'


def test_no_extrapolation_without_declared_capacity():
    d = fixture(robots=4)
    del d['profiles'][0]['capacity_bound_hz']
    p = Problem.from_dict(d)
    assert p.profiles[0].at(11).status == 'unknown'
    assert HomogeneousScheduler(p).check_frequency(3)['status'] == 'unknown'


def test_unknown_obligation_minimum_is_not_claimed_optimal():
    d = fixture('p2',gpus=3)
    d['profiles'][1] = profile('monitor',mean=100,p99=500,capacity=1)
    del d['profiles'][1]['capacity_bound_hz']
    r = HomogeneousScheduler(Problem.from_dict(d)).solve()
    assert r['status'] == 'unknown'
    assert 'footprint' in r['reason']


def test_nonmonotone_profiles_do_not_use_binary_search():
    d = fixture()
    d['profiles'][0]['points'][0]['mean_ms'] = 80
    r = HomogeneousScheduler(Problem.from_dict(d)).solve()
    assert r['status'] == 'unknown'
    assert r['search_status'] == 'unsupported_nonmonotone_profiles'


def test_a_rejected_packing_does_not_prove_infeasibility():
    p = Problem.from_dict(unresolved_packing())
    assert oracle.feasible(p,2)
    r = HomogeneousScheduler(p).check_frequency(2)
    assert r['status'] == 'unknown'
    assert r['reason'] == 'selected_packing_failed_loop_other_packings_unresolved'


def test_explicit_extension_recovers_alternative_without_relabelling_paper():
    p=Problem.from_dict(unresolved_packing())
    scheduler=HomogeneousScheduler(p,recover_alternatives=True)
    candidate=scheduler.check_frequency(2)
    assert candidate['status']=='feasible' and candidate['recovery_used']
    validate_schedule(p,2,candidate['placements'],candidate['routes'])
    result=scheduler.solve()
    assert result['algorithm_mode']=='with_assignment_recovery_extension'
    assert result['status']=='feasible'
    assert abs(result['frequency_hz']-oracle.max_frequency_grid(p))<=.01


def test_recovery_preserves_obligation_routes():
    from helpers import component
    d=unresolved_packing()
    d['hardware']['gpu_count']=4
    d['workload']['components'].append(component('monitor','obligation',2000,.5))
    d['profiles'].append(profile('monitor',mean=100,p99=500,capacity=3))
    p=Problem.from_dict(d)
    r=HomogeneousScheduler(p,recover_alternatives=True).check_frequency(2)
    assert r['status']=='feasible' and r['recovery_used']
    assert sum(x['component_id']=='monitor' for x in r['placements'])==1
    validate_schedule(p,2,r['placements'],r['routes'])


def test_zero_budget_and_empty_fleet():
    d = fixture()
    d['search']['time_limit_s'] = 0
    assert HomogeneousScheduler(Problem.from_dict(d)).solve()['status'] == 'unknown'
    d['workload']['robots'] = 0
    r = HomogeneousScheduler(Problem.from_dict(d)).solve()
    assert r['status'] == 'feasible' and r['frequency_hz'] == 0 and not r['placements']


def test_unknown_solver_status_is_not_infeasible(monkeypatch):
    from ortools.sat.python import cp_model
    monkeypatch.setattr(cp_model.CpSolver,'solve', lambda *a,**k: cp_model.UNKNOWN)
    r = HomogeneousScheduler(Problem.from_dict(fixture())).check_frequency(1)
    assert r['status'] == 'unknown'


def test_tied_solutions_are_reproducible():
    d = fixture()
    twin = copy.deepcopy(d['profiles'][0]); twin['id'] = 'same-cost-alternative'
    d['profiles'].append(twin)
    scheduler = HomogeneousScheduler(Problem.from_dict(d))
    assert scheduler.solve() == scheduler.solve()


def test_checker_detects_tampered_schedule():
    p = Problem.from_dict(fixture())
    r = HomogeneousScheduler(p).solve()
    r['routes']['0']['s1'] = 'gpu-999'
    with pytest.raises(ValueError,match='route'):
        validate_schedule(p,r['frequency_hz'],r['placements'],r['routes'])


@pytest.mark.parametrize('case', ['nan','revision','fingerprint','hardware','duplicate','unknown_key','bad_role','negative'])
def test_reject_invalid_problem(case):
    d = fixture()
    if case == 'nan': d['workload']['action_time_s'] = math.nan
    if case == 'revision': d['profiles'][0]['model_revision'] = 'wrong'
    if case == 'fingerprint': d['profiles'][0]['runtime_fingerprint'] = 'wrong'
    if case == 'hardware': d['profiles'][0]['hardware_kind'] = 'wrong'
    if case == 'duplicate': d['profiles'].append(copy.deepcopy(d['profiles'][0]))
    if case == 'unknown_key': d['hardware']['gpu_memory'] = 96
    if case == 'bad_role': d['workload']['components'][0]['role'] = 'obligation'
    if case == 'negative': d['workload']['robots'] = -1
    with pytest.raises((ValueError,ValidationError)):
        Problem.from_dict(d)


def test_profile_interpolation_and_endpoints():
    d = fixture()
    d['profiles'][0]['points'][1].update(mean_ms=100,p99_ms=180)
    p = Problem.from_dict(d).profiles[0]
    assert p.at(0).p99_ms == 100
    assert p.at(5).p99_ms == 180 and p.at(5).mean_ms == 75
    assert p.at(10).mean_ms == 100
    assert p.at(10.01).status == 'infeasible'
    assert p.at(-1).status == 'unknown'
