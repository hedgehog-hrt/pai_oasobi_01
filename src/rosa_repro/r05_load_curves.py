"""R05 single-worker fixed-point prediction on measured load domains only."""
import json,math,statistics
from copy import deepcopy
from pathlib import Path
from .execution_plan import digest,runtime_source_manifest,check_runtime_inventory
from .regression import sha256
from .profile_adoption import inside,runtime_signature
from .r05_protocol import validate_suite as validate_r05,plan_for,validate_plan as validate_r05_plan
from .profile_protocol import template,probe_plan,make_suite
from .profile_runner import reaudit

METHOD='measured_load_curve_fixed_point_v1'
PURPOSE='r05_load_curve_calibration'


def curve_plan(config,cell):
    base=plan_for(config,cell);c=template('s1','groot')
    c.update({k:deepcopy(config[k]) for k in ('mode','model_revision','runtime_fingerprint','hardware_kind','binding','batch_wait_ms')})
    c.update(streams=max(cell['robots'],cell['batch_size']),batch_size=cell['batch_size'],request_timeout_s=config['watchdog_s'],drain_timeout_s=config['drain_s'])
    p=probe_plan(c);c['streams']=cell['robots'];p['workers'][0]['members']=[f'probe-{i}' for i in range(c['streams'])]
    p.update(config=c,purpose=PURPOSE,r05_plan=base,latency_slo_ms=None,capacity_certified=False)
    p.pop('plan_id');p['plan_id']=digest(p);return p


def validate_plan(p):
    b=p['r05_plan'];validate_r05_plan(b)
    if p!=curve_plan(b['config'],b['cell']):raise ValueError('Load-curve plan differs')
    return p


def prepare(s,rates,repetitions=3):
    validate_r05(s)
    if s['stage']!='pilot' or len(s['groups'])!=1:raise ValueError('One prospective pilot group per load-curve suite required')
    if len(rates)<2:raise ValueError('At least two predeclared rates required')
    if s['config']['mode']=='gpu' and repetitions<3:raise ValueError('Three independent repetitions per rate required')
    cases=[]
    for cell in s['cells']:
        p=curve_plan(s['config'],cell);c=deepcopy(p['config']);c['streams']=max(c['streams'],c['batch_size'])
        old=make_suite(c,s['corpus'],stage='profile',phase='formal' if c['mode']=='gpu' else 'cpu_smoke',rates=rates,repetitions=repetitions)
        if not old['executable']:raise ValueError('Load-curve timing unresolved')
        for case in old['cases']:
            row=dict(case,group=cell['group'],batch_cap=cell['batch_size'],r05_cell=cell)
            row['case_id']='r05-curve-'+digest(row)[:16];cases.append(row)
    return dict(kind='r05_load_curve_suite',pilot_suite=deepcopy(s),rates=list(rates),repetitions=repetitions,cases=cases,source_manifest=runtime_source_manifest(),
        latency_slo_ms=None,diagnostic_deadline_only_ms=200,automatic_R07_adoption_allowed=False)


def validate_collection(s):
    if s!=prepare(s['pilot_suite'],s['rates'],s['repetitions']):raise ValueError('Load-curve suite modified')
    return s


def solve(points,robots,action_time_s):
    if type(robots) is not int or robots<1 or not math.isfinite(action_time_s) or action_time_s<=0:raise ValueError('Positive workload required')
    if len(points)<2:raise ValueError('At least two measured rate points required')
    points=sorted(points,key=lambda x:x['rate_hz'])
    if len({p['rate_hz'] for p in points})!=len(points) or any(type(p[k]) not in (int,float) or not math.isfinite(p[k]) or p[k]<=0 for p in points for k in ('rate_hz','mean_ms')):raise ValueError('Distinct positive measured points required')
    candidates=[]
    for left,right in zip(points,points[1:]):
        lo,hi=left['rate_hz'],right['rate_hz'];a=(right['mean_ms']-left['mean_ms'])/(hi-lo)/1000;b=action_time_s+left['mean_ms']/1000-a*lo
        xs=[lo,hi]
        if abs(a)<1e-15:
            if b>0:xs.append(robots/b)
        else:
            disc=b*b+4*a*robots
            if disc>=0:
                # Stable quadratic roots, including decreasing/nonmonotonic segments.
                q=-.5*(b+math.copysign(math.sqrt(disc),b));xs.extend([q/a,-robots/q] if q else [])
        for x in xs:
            if lo-1e-10<=x<=hi+1e-10 and x*(a*x+b)<=robots+1e-9:
                x=min(hi,max(lo,x));candidates.append(dict(rate_hz=x,mean_ms=(a*x+b-action_time_s)*1000,segment=[lo,hi]))
    if not candidates:return dict(status='unresolved',reason='no_feasible_rate_within_measured_domain',predicted_total_action_rate_hz=None)
    best=max(candidates,key=lambda p:p['rate_hz'])
    return dict(status='bounded_estimate',predicted_total_action_rate_hz=best['rate_hz'],mean_latency_ms_at_estimate=best['mean_ms'],segment=best['segment'],
        measured_domain_hz=[points[0]['rate_hz'],points[-1]['rate_hz']],right_censored=math.isclose(best['rate_hz'],points[-1]['rate_hz'],rel_tol=1e-9),
        formula='lambda = K*f; lambda*(action_time + interpolated_mean_latency(lambda)) <= K',optimality_claim=False)


def generate(root,pilot_directories,profile_directories,*,relative_tie_tolerance=.01):
    from .r05_prediction import generate as pilots
    prior=pilots(root,pilot_directories,relative_tie_tolerance=relative_tie_tolerance)
    if set(profile_directories)!=set(prior['groups']):raise ValueError('Curve/pilot groups differ')
    result=deepcopy(prior);result['method']=METHOD;result['details']={};result['groups']={};paths=set();runs=set();files=list(prior['source_evidence_sha256'])
    for group,bybatch in sorted(profile_directories.items()):
        if set(bybatch)!={'1','2','4','8','16'}:raise ValueError('All five batch curves required')
        group_rows=[];grids=[];workload=prior['details'][group]['workload']
        pilot_suite=json.loads((inside(root,pilot_directories[group][0])/'suite.json').read_text());config=pilot_suite['config'];book=pilot_suite['corpus']
        for batch,names in sorted(bybatch.items(),key=lambda item:int(item[0])):
            points={};seen=set();trials=[]
            for name in names:
                p=inside(root,name)
                if p.resolve() in paths:raise ValueError('Curve trial reused')
                paths.add(p.resolve());m=reaudit(p);read=lambda n:json.loads((p/n).read_text());x=read('condition.json');r=read('result.json');plan=validate_plan(x['plan']);cell=plan['r05_plan']['cell'];c=x['config'];case=x['case']
                if x['config']!=plan['config'] or plan['r05_plan']['config']!=config or cell['group']!=group or cell['batch_size']!=int(batch):raise ValueError('Curve model/workload/batch binding differs')
                if x['corpus_id']!=book['corpus_id'] or x['split']!='profile' or read('source_manifest.json')!=runtime_source_manifest():raise ValueError('Curve input/source differs')
                if case.get('r05_cell')!=cell or case.get('batch_cap')!=int(batch) or case.get('group')!=group:raise ValueError('Curve case binding differs')
                if case['stage']!='profile' or r['status']!='complete' or not r['drained'] or m['successful']!=m['arrivals'] or not m['successful']:raise ValueError('Failure-free completed open-loop profile required')
                if m['dispatch_lag_max_ms']>c['max_dispatch_lag_s']*1000:raise ValueError('Load-generator lag invalidates curve')
                if case['phase']!=('formal' if config['mode']=='gpu' else 'cpu_smoke'):raise ValueError('Curve phase differs')
                if any(r[phase].get('request_limit_hit') for phase in ('warmup','measurement')):raise ValueError('Curve request cap hit')
                if r['run_id'] in runs:raise ValueError('Duplicate curve run')
                runs.add(r['run_id']);rate=case['offered_hz'];rep=case['rep']
                if (rate,rep) in seen:raise ValueError('Duplicate curve rate/repetition')
                seen.add((rate,rep))
                worker=r['worker_after'];before=r['worker_before'];wid=plan['workers'][0]['worker_id']
                check_runtime_inventory(plan,{wid:worker});check_runtime_inventory(plan,{wid:before})
                if before['instance_id']!=worker['instance_id'] or before['backend_attestation']!=worker['backend_attestation']:raise ValueError('Curve worker restarted/changed')
                backend=worker['backend_attestation'];env=r['worker_environment'];signature=runtime_signature(env,backend) if config['mode']=='gpu' else dict(backend=backend,environment=env)
                if signature!=prior['runtime_signature']:raise ValueError('Curve and pilot observed runtimes differ')
                if config['mode']=='gpu':
                    from .profiling_budget import validate_formal_case
                    validate_formal_case(case,c)
                    if m['successful']<1000 or m['duration_s']<120 or r['warmup']['successful_before_cutoff']<20 or r['warmup']['duration_s']<30 or not backend.get('model_artifacts_verified'):raise ValueError('GPU curve sample/warmup/model evidence insufficient')
                points.setdefault(rate,[]).append(m['scheduled_latency_ms_successful_only']['mean'])
                h=sha256(p/'evidence.json');files.append(h);trials.append(dict(directory=name,evidence_sha256=h,rate_hz=rate,repetition=rep,actual_batch_sizes=sorted(int(k) for k in m['batch_histogram_requests'])))
            minimum=3 if config['mode']=='gpu' else 1
            if len(points)<2 or any(len(v)<minimum for v in points.values()):raise ValueError('Insufficient curve rates/repetitions')
            grid=sorted(seen);grids.append(grid)
            curve=[dict(rate_hz=rate,mean_ms=max(values),repetition_means_ms=values) for rate,values in sorted(points.items())]
            decision=solve(curve,**workload)
            if decision['status']!='bounded_estimate':raise ValueError('No supported estimate; expand measured grid')
            group_rows.append(dict(batch_cap=int(batch),points=curve,trials=trials,**decision))
        if any(g!=grids[0] for g in grids):raise ValueError('Use same predeclared rates/repetitions for every cap')
        best=max(r['predicted_total_action_rate_hz'] for r in group_rows);eligible=[r['batch_cap'] for r in group_rows if r['predicted_total_action_rate_hz']>=best*(1-relative_tie_tolerance)]
        result['groups'][group]=min(eligible);result['details'][group]=dict(workload=workload,rows=group_rows,tie_eligible_batch_caps=eligible,selected_batch_cap=min(eligible),relative_tie_tolerance=relative_tie_tolerance)
    result.update(inputs=dict(pilot_directories=pilot_directories,profile_directories=profile_directories,relative_tie_tolerance=relative_tie_tolerance),source_evidence_sha256=sorted(files),
        formula='max measured-domain lambda: lambda*(t_action + L_batch(lambda)) <= K; no latency SLO',
        limitations=['Piecewise-linear interpolation of conservative max repetition means; no smoothing, extrapolation or global optimum.',
        'Same stream count K, batch cap, runtime and corpus as pilots. Batch cap may exceed K; actual partial batches remain explicit.',
        'Open-loop inter-arrival process differs from closed-loop operation. Independent confirmation is required.',
        'Optimum at highest measured load is right-censored. Finite-window means do not prove stationary queue behavior.',
        'All five matching pilots remain operational prerequisites; their throughput does not select the curve winner.'])
    return result
