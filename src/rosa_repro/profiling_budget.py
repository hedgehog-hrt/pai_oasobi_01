"""Deterministic formal measurement budgets derived before any GPU execution.
Counts/minimum durations stay fixed; hard caps never grow during a trial.
"""
from copy import deepcopy
from decimal import Decimal,ROUND_CEILING
import math
from .contracts import obj,validate

TIMING_SCHEMA=obj(dict(schema_version={'const':'1.0'},
    headroom_factor={'type':'number','minimum':1.05,'maximum':3},
    warmup_hard_cap_s={'type':'number','minimum':180,'maximum':86400},
    measure_hard_cap_s={'type':'number','minimum':900,'maximum':172800},
    max_requests_per_phase={'type':'integer','minimum':1000,'maximum':1000000}))


def default_timing_policy():
    return dict(schema_version='1.0',headroom_factor=1.2,warmup_hard_cap_s=3600.,measure_hard_cap_s=21600.,max_requests_per_phase=100000)


def formal_timing(rate_hz,config,policy=None):
    policy=deepcopy(policy or default_timing_policy());validate(policy,TIMING_SCHEMA)
    if type(rate_hz) not in (int,float) or not math.isfinite(rate_hz) or not 1e-6<=rate_hz<=10000:raise ValueError('Offered rate must be finite and between 1e-6 and 10000 Hz')
    d=lambda x:Decimal(str(x));ceil=lambda x:int(x.to_integral_value(rounding=ROUND_CEILING))
    rate=d(rate_hz);timeout=d(config['request_timeout_s']);lag=d(config['max_dispatch_lag_s'])
    protocol={};phases={};blocked=[]
    for name,count,minimum,base_cap,hard_key in [('warmup',20,30,180,'warmup_hard_cap_s'),('measure',1000,120,900,'measure_hard_cap_s')]:
        # N/rate leaves one full inter-arrival interval beyond the Nth arrival.
        arrivals_s=d(count)/rate
        target_s=max(d(minimum),arrivals_s)+timeout+lag
        recommended=ceil(max(d(base_cap),max(d(minimum),arrivals_s)*d(policy['headroom_factor'])+timeout+lag))
        hard=d(policy[hard_key]);applied=min(recommended,float(hard))
        min_arrivals=ceil(rate*max(d(minimum),arrivals_s))
        if d(applied)<target_s:blocked.append(name+'_time_cap_cannot_cover_target_and_response_allowance')
        if policy['max_requests_per_phase']<=min_arrivals:blocked.append(name+'_request_cap_cannot_cover_target')
        protocol.update({name+'_min_s':float(minimum),name+'_min_success':count,name+'_max_s':float(applied)})
        phases[name]=dict(target_successes=count,ideal_arrival_time_s=float(arrivals_s),
            target_with_response_allowance_s=float(target_s),recommended_max_s=recommended,applied_max_s=float(applied),
            capped_by_policy=recommended>float(hard),minimum_planned_arrivals=min_arrivals,
            request_count_upper_bound=min(policy['max_requests_per_phase'],ceil(rate*d(applied))))
    protocol['max_requests']=policy['max_requests_per_phase']
    envelope=protocol['warmup_max_s']+protocol['measure_max_s']+2*(config['drain_timeout_s']+2*config['request_timeout_s'])+5
    return dict(schema_version='1.0',policy=policy,protocol=protocol,phases=phases,executable=not blocked,
        blocking_reasons=blocked,bounded_phase_and_drain_s=envelope,
        excludes='model load, file hashing, Ray scheduling, resets/environment probes and process teardown',
        success_rate_assumption='all target arrivals succeed; overload/errors can still cause shortfall',capacity_certified=False)


def validate_formal_case(case,config):
    timing=case.get('timing')
    if not isinstance(timing,dict):raise ValueError('Formal case lacks rate-derived timing evidence')
    expected=formal_timing(case['offered_hz'],config,timing['policy'])
    if timing!=expected or case['protocol']!=expected['protocol'] or not expected['executable']:
        raise ValueError('Formal timing changed or cannot fit the configured budget')
    return expected
