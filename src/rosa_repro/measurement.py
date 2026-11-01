"""R00 finalized request/action records and P1-only qualified throughput."""
from __future__ import annotations

import math
from collections import Counter

from .contracts import MEASUREMENT_SCHEMA, validate


def evaluate_p1(records, *, run_id, clock_domain, window_start_ns,
                window_end_ns, s1_component_id='s1', deadline_ms=200):
    if (type(window_start_ns) is not int or type(window_end_ns) is not int
            or window_start_ns < 0 or window_end_ns <= window_start_ns):
        raise ValueError('Explicit nonempty monotonic measurement window is required')
    if not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError('Invalid deadline')
    requests, actions = {}, []
    for record in records:
        validate(record, MEASUREMENT_SCHEMA)
        if record['run_id'] != run_id or record['clock_domain'] != clock_domain:
            raise ValueError('Mixed run or clock domain')
        if record['component_id'] != s1_component_id:
            raise ValueError('Only P1 (S1-only) is implemented; cannot qualify P2-P4')
        if record['record_type'] == 'request':
            if record['request_id'] in requests:
                raise ValueError('Duplicate request_id; each attempt needs a unique ID')
            if not record['scheduled_ns'] <= record['sent_ns'] <= record['finished_ns']:
                raise ValueError('Invalid request timestamp order')
            if record['deadline_ms'] != deadline_ms:
                raise ValueError('Mixed deadline protocol')
            if (record['batch_id'] is None) != (record['actual_batch_size'] is None):
                raise ValueError('Batch metadata must be both known or both unknown')
            if record['status'] != 'ok' and record['valid_output']:
                raise ValueError('Failed request cannot have valid_output=true')
            requests[record['request_id']] = record
        else:
            if record['end_ns'] < record['start_ns']:
                raise ValueError('Invalid action timestamp order')
            if record['applied'] != (record['executed_steps'] > 0):
                raise ValueError('Applied flag and executed steps disagree')
            actions.append(record)
    cohort = {k: r for k, r in requests.items() if not r['warmup']
              and window_start_ns <= r['sent_ns'] < window_end_ns}

    def ontime(r):
        return (r['status'] == 'ok' and r['valid_output']
                and r['finished_ns']-r['sent_ns'] <= deadline_ms*1e6)

    seen_actions, seen_requests = set(), set()
    raw, qualified, excluded = 0, 0, 0
    for a in actions:
        if a['action_id'] in seen_actions or a['request_id'] in seen_requests:
            raise ValueError('Duplicate action or repeated application of one request')
        seen_actions.add(a['action_id'])
        seen_requests.add(a['request_id'])
        r = requests.get(a['request_id'])
        if r is None:
            raise ValueError('Action has no request')
        if any(a[k] != r[k] for k in ('robot_id', 'episode_id', 'component_id')):
            raise ValueError('Action/request identity mismatch')
        if a['start_ns'] < r['finished_ns']:
            raise ValueError('Action begins before response completion')
        if a['applied'] and (r['status'] != 'ok' or not r['valid_output']):
            raise ValueError('Invalid response was applied')
        in_window = window_start_ns <= a['start_ns'] < window_end_ns
        if not in_window or not a['applied']:
            continue
        if a['request_id'] not in cohort:
            excluded += 1
            continue
        raw += 1
        qualified += int(ontime(r))
    latencies = sorted((r['finished_ns']-r['sent_ns'])/1e6 for r in cohort.values()
                       if r['status'] == 'ok' and r['valid_output'])

    def percentile(p):
        return latencies[math.ceil(len(latencies)*p)-1] if latencies else None
    duration = (window_end_ns-window_start_ns)/1e9
    meets = sum(ontime(r) for r in cohort.values())
    return {
        'schema_version': '1.0', 'metric_protocol': 'p1-finalized-send-cohort-v1',
        'run_id': run_id, 'duration_s': duration, 'deadline_ms': deadline_ms,
        'requests': len(cohort), 'request_status_counts': dict(Counter(r['status'] for r in cohort.values())),
        'deadline_qualified_requests': meets,
        'deadline_unqualified_requests': len(cohort)-meets,
        'request_deadline_meet_rate': meets/len(cohort) if cohort else None,
        'raw_action_chunks': raw, 'qualified_action_chunks': qualified,
        'raw_action_rate_hz': raw/duration, 'qualified_action_rate_hz': qualified/duration,
        'cross_boundary_or_warmup_actions_excluded': excluded,
        'latency_ms_successful_valid_only': {'p50': percentile(.5), 'p95': percentile(.95), 'p99': percentile(.99)},
        'quantile_method': 'nearest_rank', 'latency_samples': len(latencies),
        'p99_exploratory': len(latencies) < 1000,
        'full_multi_component_qualification_implemented': False,
        'window_rule': 'requests sent in [start,end); action starts in same window; finalized requests may finish later',
    }
