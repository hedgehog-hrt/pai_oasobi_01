"""Homogeneous hardware/task domain and conservative profile queries."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Literal

from .contracts import PROBLEM_SCHEMA, validate

Status = Literal["feasible", "infeasible", "unknown"]


@dataclass(frozen=True)
class Component:
    id: str
    role: str
    model_revision: str
    runtime_fingerprint: str
    deadline_ms: float
    periodic_hz: float | None = None


@dataclass(frozen=True)
class ProfilePoint:
    rate_hz: float
    mean_ms: float
    p99_ms: float
    sample_count: int | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class Estimate:
    status: Status
    mean_ms: float | None = None
    p99_ms: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Profile:
    id: str
    component_id: str
    hardware_kind: str
    model_revision: str
    runtime_fingerprint: str
    source: str
    batch_size: int
    batch_wait_ms: float
    memory_mib: float
    points: tuple[ProfilePoint, ...]
    capacity_bound_hz: float | None = None
    evidence_sha256: str | None = None

    @property
    def monotone(self):
        return all(a.mean_ms <= b.mean_ms and a.p99_ms <= b.p99_ms
                   for a, b in zip(self.points, self.points[1:]))

    def at(self, rate_hz: float) -> Estimate:
        if self.capacity_bound_hz is not None and rate_hz > self.capacity_bound_hz:
            return Estimate("infeasible", reason="declared_capacity_bound")
        rates = [p.rate_hz for p in self.points]
        if rate_hz < rates[0] or rate_hz > rates[-1]:
            return Estimate("unknown", reason="unprofiled_rate")
        i = bisect.bisect_left(rates, rate_hz)
        if rates[i] == rate_hz:
            p = self.points[i]
            return Estimate("feasible", p.mean_ms, p.p99_ms)
        a, b = self.points[i-1:i+1]
        fraction = (rate_hz-a.rate_hz)/(b.rate_hz-a.rate_hz)
        return Estimate("feasible", a.mean_ms + fraction*(b.mean_ms-a.mean_ms),
                        max(a.p99_ms, b.p99_ms), "interpolated_mean_upper_endpoint_p99")


@dataclass(frozen=True)
class Problem:
    workload_id: str
    robots: int
    action_time_s: float
    s2_horizon: int
    components: tuple[Component, ...]
    hardware_kind: str
    gpu_count: int
    memory_mib: float
    profiles: tuple[Profile, ...]
    tolerance_hz: float
    time_limit_s: float

    @classmethod
    def from_dict(cls, data):
        validate(data, PROBLEM_SCHEMA)
        w, h, search = data['workload'], data['hardware'], data['search']
        components = tuple(Component(**c) for c in w['components'])
        ids = [c.id for c in components]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate component IDs")
        if sum(c.role == 's1' for c in components) != 1:
            raise ValueError("Exactly one S1 component is required")
        if sum(c.role == 's2' for c in components) > 1:
            raise ValueError("At most one S2 component is supported")
        for c in components:
            if (c.role == 'obligation') != (c.periodic_hz is not None):
                raise ValueError("Only obligations must specify periodic_hz")
        profiles = []
        for raw in data['profiles']:
            raw = dict(raw)
            raw['points'] = tuple(ProfilePoint(**v) for v in raw['points'])
            p = Profile(**raw)
            if p.component_id not in ids:
                raise ValueError(f"Unknown component in profile {p.id}")
            c = next(c for c in components if c.id == p.component_id)
            if (p.hardware_kind, p.model_revision, p.runtime_fingerprint) != (
                    h['kind'], c.model_revision, c.runtime_fingerprint):
                raise ValueError(f"Profile fingerprint mismatch: {p.id}")
            rates = [x.rate_hz for x in p.points]
            if any(a >= b for a, b in zip(rates, rates[1:])):
                raise ValueError(f"Profile rates must be strictly increasing: {p.id}")
            if p.source == 'measured' and rates[0] == 0:
                raise ValueError("Zero-rate measured points are not actual measurements")
            if p.source == 'measured' and (p.evidence_sha256 is None or any(
                    x.sample_count is None or x.duration_s is None for x in p.points)):
                raise ValueError('Measured profile needs evidence hash, sample counts, and durations')
            if p.capacity_bound_hz is not None and p.capacity_bound_hz < rates[-1]:
                raise ValueError("Capacity bound is below a profiled point")
            if c.role != 's1' and p.batch_size != 1:
                raise ValueError("S2/obligation use continuous batching notation b=1")
            profiles.append(p)
        if len({p.id for p in profiles}) != len(profiles):
            raise ValueError("Duplicate profile IDs")
        return cls(w['id'], w['robots'], w['action_time_s'], w['s2_horizon'],
                   components, h['kind'], h['gpu_count'], h['memory_mib'],
                   tuple(sorted(profiles, key=lambda p: p.id)),
                   search['frequency_tolerance_hz'], search['time_limit_s'])

    def component(self, component_id):
        return next(c for c in self.components if c.id == component_id)


@dataclass(frozen=True)
class Configuration:
    component_id: str
    profile_id: str
    streams: int
    rate_hz: float
    batch_size: int
    batch_wait_ms: float
    memory_mib: float
    mean_ms: float
    p99_ms: float

    @property
    def key(self):
        return self.component_id, self.profile_id, self.streams
