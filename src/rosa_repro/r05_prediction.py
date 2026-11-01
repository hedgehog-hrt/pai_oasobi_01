"""Prospective batch-cap selection from measured, workload-matched pilot means.

The single-S1 specialization of paper Eq. (4) is K/(t_action + mean_latency).
It is a closed-loop mean-cycle estimate at the measured workload, not a fitted
open-loop load curve, a global optimum, or a confidence bound. Confirmation
outcomes never enter this selector.
"""
import json,math,statistics
from pathlib import Path
from .execution_plan import digest,runtime_source_manifest
from .regression import sha256
from .profile_adoption import inside,runtime_signature
from .r05_protocol import validate_suite
from .r05_runtime import reaudit

METHOD='measured_closed_loop_mean_cycle_v1'


def rank(points,*,robots,action_time_s,relative_tie_tolerance=.01):
    if type(robots) is not int or robots<1 or not math.isfinite(action_time_s) or action_time_s<=0 or not 0<=relative_tie_tolerance<=.05:raise ValueError('Invalid prediction controls')
    if {p['batch_cap'] for p in points}!={1,2,4,8,16} or len(points)!=5:raise ValueError('Exactly five measured batch caps required')
    rows=[]
    for p in points:
        if type(p['mean_latency_ms']) not in (int,float) or not math.isfinite(p['mean_latency_ms']) or p['mean_latency_ms']<=0:raise ValueError('Positive measured latency required')
        rows.append(dict(p,predicted_total_action_rate_hz=robots/(action_time_s+p['mean_latency_ms']/1000)))
    best=max(r['predicted_total_action_rate_hz'] for r in rows)
    eligible=[r for r in rows if r['predicted_total_action_rate_hz']>=best*(1-relative_tie_tolerance)]
    selected=min(r['batch_cap'] for r in eligible)
    return dict(selected_batch_cap=selected,rows=sorted(rows,key=lambda r:r['batch_cap']),
        maximum_point_estimate_hz=best,tie_eligible_batch_caps=sorted(r['batch_cap'] for r in eligible),relative_tie_tolerance=relative_tie_tolerance)


def generate(root,pilot_directories,*,relative_tie_tolerance=.01):
    if not isinstance(pilot_directories,dict) or not pilot_directories:raise ValueError('Pilot directory map required')
    groups={};details={};files=[];paths=set();runs=set();binding=None;runtime=None;source=runtime_source_manifest()
    for group,names in sorted(pilot_directories.items()):
        if not isinstance(names,list) or len(names)!=5:raise ValueError('Exactly five pilot paths per group required')
        points=[];workload=None;controls=None
        for name in names:
            p=inside(root,name)
            if p.resolve() in paths:raise ValueError('Pilot reused across predictions')
            paths.add(p.resolve());audit=reaudit(p)
            read=lambda n:json.loads((p/n).read_text())
            s=validate_suite(read('suite.json'));cell=audit['cell'];r=read('result.json')
            if audit['status']!='complete' or s['stage']!='pilot' or cell['group']!=group:raise ValueError('Complete matching pilot required; confirmation cannot select a batch')
            if r['run_id'] in runs:raise ValueError('Duplicate pilot run')
            runs.add(r['run_id'])
            bound={k:s[k] for k in ('config','corpus','split','source_manifest')}
            if binding is not None and bound!=binding:raise ValueError('Prediction input/runtime/source differs')
            binding=bound
            current_controls=dict(protocol=s['protocol'],trial_seed=cell['trial_seed'])
            if controls is not None and current_controls!=controls:raise ValueError('Pilot timing/seed controls differ within group')
            controls=current_controls
            current_workload=dict(robots=cell['robots'],action_time_s=cell['action_time_s'])
            if workload is not None and current_workload!=workload:raise ValueError('Pilot workload differs')
            workload=current_workload
            backend=audit['backend_attestation'];env=read('worker_environment.json')
            signature=runtime_signature(env,backend) if s['config']['mode']=='gpu' else dict(backend=backend,environment=env)
            if runtime is not None and signature!=runtime:raise ValueError('Observed pilot runtime differs')
            runtime=signature
            if s['config']['mode']=='gpu' and (backend.get('execution_kind')!='external_model' or backend.get('device_type')!='cuda'):raise ValueError('GPU model execution evidence required')
            rows=list(map(json.loads,(p/'events.jsonl').read_text().splitlines()))
            starts={e['request_id']:e for e in rows if e['kind']=='request_start' and e['phase']=='measurement'}
            ends=[e for e in rows if e['kind']=='request_end' and e['phase']=='measurement']
            if len(ends)!=len(starts) or any(e['status']!='ok' for e in ends) or not ends:raise ValueError('Complete failure-free measured latency cohort required')
            latencies=[(e['time_ns']-starts[e['request_id']]['time_ns'])/1e6 for e in ends]
            h=sha256(p/'evidence.json');files.append(h)
            points.append(dict(batch_cap=cell['batch_size'],mean_latency_ms=statistics.mean(latencies),requests=len(ends),
                actual_batch_sizes=sorted({e['actual_batch_size'] for e in ends}),directory=name,evidence_file_sha256=h,
                observed_completed_action_rate_hz=r['metrics']['measurement']['completed_virtual_action_rate_hz']))
        decision=rank(points,**workload,relative_tie_tolerance=relative_tie_tolerance)
        groups[group]=decision['selected_batch_cap'];details[group]=dict(workload=workload,controls=controls,**decision)
    return dict(schema_version='1.0',kind='r05_batch_prediction',method=METHOD,mode=binding['config']['mode'],
        config_sha256=digest(binding['config']),corpus_id=binding['corpus']['corpus_id'],source_manifest=source,
        groups=groups,details=details,source_evidence_sha256=sorted(files),runtime_signature=runtime,
        inputs=dict(pilot_directories=pilot_directories,relative_tie_tolerance=relative_tie_tolerance),
        formula='total_action_rate_hz = robots / (action_time_s + mean_response_latency_ms / 1000)',
        latency_slo_ms=None,optimality_claim=False,independent_confirmation_required=True,
        limitations=['Workload-specific single-worker mean-cycle estimate; no unmeasured batch, robot count or action duration extrapolation.',
          'Queue, transport and batch wait are included in measured response latency. CUDA event duration is not substituted.',
          'Finite pilot windows, boundary completions and stochastic latency can bias predictions; no confidence interval or global optimum is claimed.',
          'Smallest measured batch cap within the predeclared relative tie tolerance wins. Partial observed batches remain explicit.',
          'This does not implement a multi-worker placement search or reproduce the paper hardware. Confirmation repetitions remain independent.'])


def verify(document,root):
    if document.get('method') not in (METHOD,'measured_load_curve_fixed_point_v1') or document.get('source_manifest')!=runtime_source_manifest():raise ValueError('Generated current-source prediction required')
    if document['method']==METHOD:expected=generate(root,**document['inputs'])
    else:
        from .r05_load_curves import generate as from_curves
        expected=from_curves(root,**document['inputs'])
    if document!=expected:raise ValueError('Prediction differs from recomputed measured evidence')
    return document
