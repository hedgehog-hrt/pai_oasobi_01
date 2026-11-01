"""ROSA §3.4.2 independent implementation with explicit uncertainty.

Only homogeneous hardware and homogeneous tasks are implemented. OR-Tools
CP-SAT solves integer linear packing constraints (paper Eq.3). No GPU calls.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from ortools.sat.python import cp_model

from .domain import Configuration, Problem


@dataclass
class Budget:
    seconds: float
    started: float = field(default_factory=time.monotonic)

    def remaining(self):
        return max(0., self.seconds-(time.monotonic()-self.started))


@dataclass
class Packing:
    status: str
    selected: list[Configuration] = field(default_factory=list)
    optimal: bool = False
    reason: str = ""


def enumerate_configurations(problem, components, frequency, budget):
    configs, unknown = [], []
    for c in components:
        profiles = [p for p in problem.profiles if p.component_id == c.id]
        if not profiles:
            unknown.append(f"missing_profile:{c.id}")
        rate = c.periodic_hz if c.role == 'obligation' else (
            frequency/problem.s2_horizon if c.role == 's2' else frequency)
        for p in profiles:
            if p.memory_mib > problem.memory_mib:
                continue
            for k in range(1, problem.robots+1):
                if not budget.remaining():
                    return configs, unknown + ['time_budget']
                estimate = p.at(k*rate)
                if estimate.status == 'unknown':
                    unknown.append(f"unprofiled:{p.id}:{k*rate:.9g}")
                if estimate.status != 'feasible' or estimate.p99_ms > c.deadline_ms:
                    continue
                configs.append(Configuration(c.id, p.id, k, k*rate, p.batch_size,
                                              p.batch_wait_ms, p.memory_mib,
                                              estimate.mean_ms, estimate.p99_ms))
    return sorted(configs, key=lambda x: x.key), unknown


def pack(configs, components, robots, gpu_budget, budget, *, obligations=False):
    if robots == 0 or not components:
        return Packing('feasible', optimal=True)
    if gpu_budget < len(components):
        return Packing('infeasible', reason='one_gpu_per_required_component')
    model = cp_model.CpModel()
    y = [model.new_int_var(0, gpu_budget, f'config_{i}') for i in range(len(configs))]
    model.add(sum(y) <= gpu_budget)
    for c in components:
        model.add(sum(v*p.streams for v, p in zip(y, configs) if p.component_id == c.id) == robots)
    # Quantization is only for tie-breaking, never latency feasibility.
    latency_cost = sum(v*p.streams*math.ceil(p.mean_ms*1000) for v, p in zip(y, configs))
    server_cost = sum(y)
    stable_cost = sum((i+1)*v for i, v in enumerate(y))
    objectives = [server_cost, latency_cost, stable_cost] if obligations else [latency_cost, server_cost, stable_cost]
    selected = []
    for objective in objectives:
        if not budget.remaining():
            return Packing('feasible' if selected else 'unknown', selected, False, 'time_budget')
        model.minimize(objective)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = budget.remaining()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(f"Invalid ILP: {model.validate()}")
        if status == cp_model.INFEASIBLE:
            return Packing('infeasible', reason='integer_cover_infeasible')
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Packing('feasible' if selected else 'unknown', selected, False, 'solver_unknown')
        selected = [p for v, p in zip(y, configs) for _ in range(solver.value(v))]
        if status != cp_model.OPTIMAL:
            return Packing('feasible', selected, False, 'tie_break_not_proven')
        model.add(objective == solver.value(objective))
    return Packing('feasible', selected, True)


def route(problem, selected):
    placements, routes = [], {str(i): {} for i in range(problem.robots)}
    next_robot = {c.id: 0 for c in problem.components}
    for index, conf in enumerate(sorted(selected, key=lambda x: x.key)):
        worker = f'gpu-{index}'
        start = next_robot[conf.component_id]
        robot_ids = [str(i) for i in range(start, start+conf.streams)]
        next_robot[conf.component_id] += conf.streams
        placements.append({'worker_id': worker, 'robot_ids': robot_ids, **asdict(conf)})
        for rid in robot_ids:
            routes[rid][conf.component_id] = worker
    return placements, routes


def average_loop_relaxation_infeasible(problem, configs, goals, gpu_budget, frequency, budget):
    """Prove impossibility only when a necessary average-loop bound fails.

    Every per-robot feasible schedule satisfies this relaxed constraint. A
    feasible relaxation does NOT prove a feasible routing. Round latency down
    and budget up, so quantization cannot create a false infeasibility proof.
    """
    if not budget.remaining():
        return False
    model = cp_model.CpModel()
    y = [model.new_int_var(0, gpu_budget, f'bound_{i}') for i in range(len(configs))]
    model.add(sum(y) <= gpu_budget)
    for c in goals:
        model.add(sum(v*x.streams for v,x in zip(y,configs) if x.component_id==c.id)==problem.robots)
    h = problem.s2_horizon
    terms = []
    for v,x in zip(y,configs):
        weight = h if problem.component(x.component_id).role=='s1' else 1
        terms.append(v*x.streams*math.floor(x.mean_ms*1000)*weight)
    rhs = math.ceil((1/frequency-problem.action_time_s)*1e6*problem.robots*h)+1
    model.add(sum(terms) <= rhs)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget.remaining()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError(f'Invalid relaxed ILP: {model.validate()}')
    return status == cp_model.INFEASIBLE


def recover_alternative(problem, configs, goals, obligations, frequency, budget):
    """Optional extension: explicitly assign robot streams to configurations.

    Bounded to small R01 instances. This is NOT the paper's published packing
    pseudocode and is reported separately. Only a validated positive witness
    is used; rounded-model infeasibility is deliberately not a global proof.
    """
    available = problem.gpu_count-len(obligations)
    if problem.robots > 8 or available > 8:
        return {'status':'unknown','reason':'recovery_small_instance_limit'}
    if not budget.remaining():
        return {'status':'unknown','reason':'time_budget'}
    model=cp_model.CpModel()
    active, assignment = {}, {}
    for gpu in range(available):
        for i,c in enumerate(configs):
            active[gpu,i]=model.new_bool_var(f'use_{gpu}_{i}')
            for robot in range(problem.robots):
                z=model.new_bool_var(f'route_{gpu}_{i}_{robot}')
                assignment[gpu,i,robot]=z
                model.add(z<=active[gpu,i])
            model.add(sum(assignment[gpu,i,r] for r in range(problem.robots))==c.streams*active[gpu,i])
        model.add(sum(active[gpu,i] for i in range(len(configs)))<=1)
    for r in range(problem.robots):
        for c in goals:
            model.add(sum(assignment[g,i,r] for g in range(available)
                          for i,x in enumerate(configs) if x.component_id==c.id)==1)
        h=problem.s2_horizon
        terms=[]
        for g in range(available):
            for i,c in enumerate(configs):
                weight=h if problem.component(c.component_id).role=='s1' else 1
                terms.append(assignment[g,i,r]*math.ceil(c.mean_ms*1000)*weight)
        rhs=math.floor((1/frequency-problem.action_time_s)*1e6*h+1e-6)
        model.add(sum(terms)<=rhs)
    # A feasibility witness is enough for this recovery; optimization remains
    # the outer rate search. Prefer fewer workers as a secondary objective.
    model.minimize(sum(active.values()))
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=budget.remaining()
    solver.parameters.num_search_workers=1
    solver.parameters.random_seed=0
    status=solver.solve(model)
    if status==cp_model.MODEL_INVALID:
        raise RuntimeError(f'Invalid recovery ILP: {model.validate()}')
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        return {'status':'unknown','reason':'alternative_search_no_validated_witness'}
    placements,routes=route(problem,obligations)
    for gpu in range(available):
        for i,c in enumerate(configs):
            if not solver.value(active[gpu,i]):
                continue
            worker=f'gpu-{len(obligations)+gpu}'
            ids=[str(r) for r in range(problem.robots) if solver.value(assignment[gpu,i,r])]
            placements.append({'worker_id':worker,'robot_ids':ids,**asdict(c)})
            for rid in ids:
                routes[rid][c.component_id]=worker
    validate_schedule(problem,frequency,placements,routes)
    return {'status':'feasible','frequency_hz':frequency,'placements':placements,'routes':routes,
            'packing_optimal':status==cp_model.OPTIMAL,'recovery_used':True,
            'recovery_method':'per_robot_assignment_extension_not_paper_pseudocode'}


def validate_schedule(problem: Problem, frequency: float, placements, routes):
    """Independent checker of a returned witness; never trusts solver totals."""
    if not math.isfinite(frequency) or frequency < 0:
        raise ValueError('Invalid frequency')
    if problem.robots and frequency == 0:
        raise ValueError('Nonempty fleet requires positive frequency')
    if len(placements) > problem.gpu_count:
        raise ValueError('GPU budget exceeded')
    workers = {p['worker_id']: p for p in placements}
    if len(workers) != len(placements):
        raise ValueError('Duplicate worker IDs')
    if set(routes) != {str(i) for i in range(problem.robots)}:
        raise ValueError('Robot coverage mismatch')
    expected = {c.id for c in problem.components}
    for p in placements:
        c = problem.component(p['component_id'])
        profile = next(x for x in problem.profiles if x.id == p['profile_id'])
        if profile.component_id != c.id:
            raise ValueError('Wrong profile component')
        if len(p['robot_ids']) != p['streams'] or len(set(p['robot_ids'])) != p['streams']:
            raise ValueError('Invalid stream count')
        per_robot = c.periodic_hz if c.role == 'obligation' else frequency/(problem.s2_horizon if c.role == 's2' else 1)
        estimate = profile.at(p['streams']*per_robot)
        if estimate.status != 'feasible' or estimate.p99_ms > c.deadline_ms:
            raise ValueError('Unqualified profile')
        if profile.memory_mib > problem.memory_mib:
            raise ValueError('Memory exceeded')
        if (p['mean_ms'], p['p99_ms'], p['batch_size'], p['batch_wait_ms'], p['memory_mib']) != (
                estimate.mean_ms, estimate.p99_ms, profile.batch_size, profile.batch_wait_ms, profile.memory_mib):
            raise ValueError('Profile values changed')
        if not math.isclose(p['rate_hz'], p['streams']*per_robot, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Rate mismatch')
        for rid in p['robot_ids']:
            if rid not in routes or routes[rid].get(c.id) != p['worker_id']:
                raise ValueError('Invalid forward route')
    for rid, mapping in routes.items():
        if set(mapping) != expected:
            raise ValueError('Required component missing or duplicated')
        loop_s = problem.action_time_s
        for c in problem.components:
            p = workers.get(mapping[c.id])
            if p is None or p['component_id'] != c.id or rid not in p['robot_ids']:
                raise ValueError('Invalid reverse route')
            if c.role in ('s1', 's2'):
                loop_s += p['mean_ms']/1000/(problem.s2_horizon if c.role == 's2' else 1)
        if frequency*loop_s > 1+1e-12:
            raise ValueError('Closed-loop rate exceeded')


class HomogeneousScheduler:
    def __init__(self, problem, *, recover_alternatives=False):
        self.problem = problem
        self.recover_alternatives = recover_alternatives

    def provision(self, budget):
        p = self.problem
        obs = [c for c in p.components if c.role == 'obligation']
        configs, unknown = enumerate_configurations(p, obs, 0, budget)
        result = pack(configs, obs, p.robots, p.gpu_count, budget, obligations=True)
        if result.status == 'infeasible' and unknown:
            result.status, result.reason = 'unknown', 'obligation_profile_incomplete'
        if result.status == 'feasible' and obs and p.robots:
            if not result.optimal:
                return Packing('unknown', reason='minimum_obligation_footprint_unproven')
            if unknown and len(result.selected) > len(obs):
                return Packing('unknown', reason='unknown_profiles_may_reduce_obligation_footprint')
        return result

    def check_frequency(self, frequency, budget=None, provisioned=None):
        p = self.problem
        budget = budget or Budget(p.time_limit_s)
        if not math.isfinite(frequency) or frequency < 0:
            raise ValueError('Frequency must be finite and nonnegative')
        if p.robots == 0:
            return {'status': 'feasible', 'frequency_hz': 0., 'placements': [], 'routes': {}, 'packing_optimal': True}
        if frequency == 0:
            return {'status': 'unknown', 'reason': 'zero_rate_is_search_sentinel'}
        if frequency > 1/p.action_time_s:
            return {'status': 'infeasible', 'reason': 'action_time_upper_bound'}
        if p.gpu_count < len(p.components):
            return {'status': 'infeasible', 'reason': 'one_gpu_per_required_component'}
        if not budget.remaining():
            return {'status': 'unknown', 'reason': 'time_budget'}
        obligation = provisioned if provisioned is not None else self.provision(budget)
        if obligation.status != 'feasible':
            return {'status': obligation.status, 'reason': obligation.reason}
        goals = [c for c in p.components if c.role != 'obligation']
        configs, unknown = enumerate_configurations(p, goals, frequency, budget)
        remaining = p.gpu_count-len(obligation.selected)
        packed = pack(configs, goals, p.robots, remaining, budget)
        if packed.status != 'feasible':
            return {'status': 'unknown' if unknown else packed.status,
                    'reason': 'goal_profile_incomplete' if unknown else packed.reason}
        placements, routes = route(p, obligation.selected+packed.selected)
        try:
            validate_schedule(p, frequency, placements, routes)
        except ValueError as exc:
            if str(exc) != 'Closed-loop rate exceeded':
                raise
            # A single rejected packing does NOT prove that all packings fail.
            optimistic_loop = p.action_time_s
            for c in goals:
                optimistic_loop += min(x.mean_ms for x in configs if x.component_id == c.id)/1000/(
                    p.s2_horizon if c.role == 's2' else 1)
            impossible = not unknown and (frequency*optimistic_loop > 1+1e-12 or
                average_loop_relaxation_infeasible(p, configs, goals, remaining, frequency, budget))
            if not impossible and self.recover_alternatives:
                return recover_alternative(p,configs,goals,obligation.selected,frequency,budget)
            return {'status': 'infeasible' if impossible else 'unknown',
                    'reason': 'necessary_closed_loop_bound' if impossible else 'selected_packing_failed_loop_other_packings_unresolved'}
        return {'status': 'feasible', 'frequency_hz': frequency, 'placements': placements,
                'routes': routes, 'packing_optimal': packed.optimal,
                'unknown_profile_queries': len(unknown)}

    def solve(self):
        p = self.problem
        budget = Budget(p.time_limit_s)
        base = {'schema_version': '1.0', 'algorithm': 'rosa-v1-homogeneous-independent',
                'algorithm_mode': 'with_assignment_recovery_extension' if self.recover_alternatives else 'paper_first',
                'profile_sources': sorted({x.source for x in p.profiles}),
                'gpu_execution': False, 'workload_id': p.workload_id,
                'tolerance_hz': p.tolerance_hz, 'trace': []}
        if not p.robots:
            return {**base, 'status': 'feasible', 'search_status': 'empty_fleet',
                    'frequency_hz': 0., 'placements': [], 'routes': {}, 'rate_interval_hz': [0., 0.]}
        if p.gpu_count < len(p.components):
            return {**base, 'status': 'infeasible', 'search_status': 'infeasible',
                    'reason': 'one_gpu_per_required_component', 'frequency_hz': None}
        if any(not profile.monotone for profile in p.profiles):
            return {**base, 'status': 'unknown', 'search_status': 'unsupported_nonmonotone_profiles',
                    'reason': 'binary_search_precondition_failed', 'frequency_hz': None}
        obligation = self.provision(budget)
        if obligation.status != 'feasible':
            return {**base, 'status': obligation.status, 'search_status': obligation.status,
                    'reason': obligation.reason, 'frequency_hz': None}
        lo, hi, best = 0., 1/p.action_time_s, None
        termination = 'converged'
        while hi-lo > p.tolerance_hz:
            if not budget.remaining():
                termination = 'time_budget'
                break
            f = (lo+hi)/2
            candidate = self.check_frequency(f, budget, obligation)
            base['trace'].append({'frequency_hz': f, 'status': candidate['status'],
                                  'reason': candidate.get('reason', ''),
                                  'recovery_used': candidate.get('recovery_used',False)})
            if candidate['status'] == 'feasible':
                lo, best = f, candidate
            elif candidate['status'] == 'infeasible':
                hi = f
            else:
                termination = candidate.get('reason', 'unknown')
                break
        if best is None:
            # No positive witness in the tolerance interval is NOT a proof of
            # infeasibility for all arbitrarily small positive frequencies.
            return {**base, 'status': 'unknown', 'search_status': termination,
                    'reason': 'no_positive_witness', 'frequency_hz': None,
                    'rate_interval_hz': [lo, hi]}
        return {**base, **best, 'search_status': termination,
                'solution_quality': 'epsilon_bracket_under_declared_model' if termination == 'converged' else 'feasible_lower_bound',
                'rate_interval_hz': [lo, hi], 'predicted_action_rate_hz': p.robots*lo}
