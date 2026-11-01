"""Independent exhaustive small-instance oracle, not a production solver.

Does not call production profile lookup, configuration generation, routing,
packing, or frequency search. Enumerates all known-profile stream partitions.
"""
from __future__ import annotations

import itertools
import math


def _sample(points, rate, capacity):
    if capacity is not None and rate > capacity:
        return None
    for p in points:
        if p.rate_hz == rate:
            return p.mean_ms, p.p99_ms
    for left, right in zip(points, points[1:]):
        if left.rate_hz < rate < right.rate_hz:
            weight = (rate-left.rate_hz)/(right.rate_hz-left.rate_hz)
            return ((1-weight)*left.mean_ms+weight*right.mean_ms,
                    max(left.p99_ms, right.p99_ms))
    return None


def _component_partitions(problem, c, f):
    per_robot = c.periodic_hz if c.role == 'obligation' else f/(
        problem.s2_horizon if c.role == 's2' else 1)
    options = []
    for p in problem.profiles:
        if p.component_id != c.id or p.memory_mib > problem.memory_mib:
            continue
        for k in range(1, problem.robots+1):
            sample = _sample(p.points, per_robot*k, p.capacity_bound_hz)
            if sample is not None and sample[1] <= c.deadline_ms:
                options.append((k, sample[0], p.id))
    answers = []

    def visit(remaining, first, selected):
        if remaining == 0:
            answers.append(tuple(selected))
            return
        if len(selected) == problem.gpu_count:
            return
        for index in range(first, len(options)):
            option = options[index]
            if option[0] <= remaining:
                visit(remaining-option[0], index, selected+[option])
    visit(problem.robots, 0, [])
    return answers


def feasible(problem, frequency):
    if problem.robots > 6 or problem.gpu_count > 8:
        raise ValueError('Oracle is bounded to 6 robots and 8 logical GPUs')
    if not problem.robots:
        return True
    if frequency <= 0 or frequency > 1/problem.action_time_s:
        return False
    choices = [_component_partitions(problem, c, frequency) for c in problem.components]
    if any(not x for x in choices):
        return False
    obligation_indices = [i for i, c in enumerate(problem.components) if c.role == 'obligation']
    # Same two-stage domain as the paper: minimum obligation footprint first.
    obligation_minimum = sum(min(len(x) for x in choices[i]) for i in obligation_indices)
    for allocation in itertools.product(*choices):
        if sum(map(len, allocation)) > problem.gpu_count:
            continue
        if sum(len(allocation[i]) for i in obligation_indices) != obligation_minimum:
            continue
        s1, s2 = [], [0.]*problem.robots
        for c, selected in zip(problem.components, allocation):
            means = [option[1] for option in selected for _ in range(option[0])]
            if c.role == 's1':
                s1 = means
            elif c.role == 's2':
                s2 = [x/problem.s2_horizon for x in means]
        # Oppositely sorted pairing minimizes the maximum pair sum. The
        # production implementation uses ID-order routing, so this oracle can
        # expose routing-induced false infeasibility rather than copy it.
        loop = max(problem.action_time_s+(a+b)/1000
                   for a, b in zip(sorted(s1), sorted(s2, reverse=True)))
        if frequency*loop <= 1+1e-12:
            return True
    return False


def max_frequency_grid(problem, step_hz=.005):
    if not math.isfinite(step_hz) or step_hz <= 0:
        raise ValueError('step_hz must be positive and finite')
    if not problem.robots:
        return 0.
    for i in range(math.floor((1/problem.action_time_s)/step_hz), 0, -1):
        f = i*step_hz
        if feasible(problem, f):
            return f
    return 0.
