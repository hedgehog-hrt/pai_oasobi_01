import copy


def component(cid='s1', role='s1', deadline=200, rate=None):
    c = {'id': cid, 'role': role, 'model_revision': 'synthetic-v1',
         'runtime_fingerprint': 'fixture-v1', 'deadline_ms': deadline}
    if rate is not None:
        c['periodic_hz'] = rate
    return c


def profile(cid='s1', mean=50, p99=100, capacity=20, batch=1, memory=1000, pid=None):
    return {'id': pid or f'{cid}-b{batch}', 'component_id': cid,
            'hardware_kind': 'synthetic-gpu', 'model_revision': 'synthetic-v1',
            'runtime_fingerprint': 'fixture-v1', 'source': 'synthetic',
            'batch_size': batch, 'batch_wait_ms': 0, 'memory_mib': memory,
            'capacity_bound_hz': capacity,
            'points': [{'rate_hz': 0, 'mean_ms': mean, 'p99_ms': p99},
                       {'rate_hz': capacity, 'mean_ms': mean, 'p99_ms': p99}]}


def fixture(name='s1', robots=4, gpus=None):
    c, profiles = [component()], [profile(capacity=10)]
    if name == 'batch_tradeoff':
        profiles += [profile(mean=80, p99=150, capacity=30, batch=4)]
    if name in ('s1_s2', 'p4'):
        profiles[0] = profile(capacity=40)
        c.append(component('s2', 's2', 2000))
        profiles.append(profile('s2', mean=500, p99=1000, capacity=5))
    if name == 'p4':
        c.append(component('safety', 'obligation', 500, 2))
        profiles.append(profile('safety', mean=100, p99=300, capacity=10))
    if name in ('p2', 'p4'):
        c.append(component('monitor', 'obligation', 2000, .5))
        profiles.append(profile('monitor', mean=100, p99=500, capacity=3))
    return {'schema_version': '1.0',
            'workload': {'id': f'{name}-synthetic', 'robots': robots, 'action_time_s': .2,
                         's2_horizon': 10, 'components': c},
            'hardware': {'kind': 'synthetic-gpu', 'gpu_count': gpus if gpus is not None else len(c),
                         'memory_mib': 24000},
            'profiles': profiles,
            'search': {'frequency_tolerance_hz': .01, 'time_limit_s': 60}}


def unresolved_packing():
    d = fixture('s1_s2', robots=4, gpus=3)
    d['workload']['action_time_s'] = .31
    d['workload']['components'][0]['deadline_ms'] = 500
    a = profile(mean=0, p99=50, capacity=6, pid='s1-load-dependent')
    a['points'] = [
        {'rate_hz': 0, 'mean_ms': 0, 'p99_ms': 50},
        {'rate_hz': 2, 'mean_ms': 0, 'p99_ms': 50},
        {'rate_hz': 4, 'mean_ms': 200, 'p99_ms': 300},
        {'rate_hz': 6, 'mean_ms': 200, 'p99_ms': 300},
    ]
    d['profiles'] = [a, profile(mean=180, p99=250, capacity=4, batch=2),
                     profile('s2', mean=0, p99=50, capacity=6)]
    return d


def request(rid='q1', *, sent=0, duration_ms=50, status='ok', valid=True, warmup=False):
    return {'schema_version': '1.0', 'record_type': 'request',
            'run_id': 'run', 'robot_id': 'r1', 'episode_id': 'e1',
            'request_id': rid, 'component_id': 's1', 'attempt_id': f'{rid}-attempt0',
            'observation_seq': 0, 'clock_domain': 'host-boot-1',
            'scheduled_ns': sent, 'sent_ns': sent, 'finished_ns': sent+int(duration_ms*1e6),
            'deadline_ms': 200, 'status': status, 'valid_output': valid,
            'warmup': warmup, 'batch_id': None, 'actual_batch_size': None}


def action(r, *, start=None, steps=8):
    return {'schema_version': '1.0', 'record_type': 'action',
            **{k:r[k] for k in ('run_id','robot_id','episode_id','request_id','component_id','clock_domain')},
            'action_id': f"action-{r['request_id']}",
            'start_ns': r['finished_ns'] if start is None else start,
            'end_ns': (r['finished_ns'] if start is None else start)+400_000_000,
            'applied': steps > 0, 'executed_steps': steps}
