"""Prospective dynamic trial admission. Transients remain outcomes, not filtered samples."""
from copy import deepcopy
import math
from .execution_plan import digest


def protocol(spec,*,phase,warmup_s,duration_s,warmup_min_success=None,measure_min_success=1,change_tolerance_s=1.):
    if phase not in ('cpu_smoke','formal') or (phase=='formal')!=(spec['catalog']['mode']=='gpu'):raise ValueError('Formal dynamic protocol requires GPU; CPU uses smoke')
    minimum=20 if phase=='formal' else 1;warm=minimum if warmup_min_success is None else warmup_min_success
    if type(warm) is not int or not minimum<=warm<=10000 or type(measure_min_success) is not int or not 1<=measure_min_success<=100000:raise ValueError('Invalid sample gates')
    if any(type(x) not in (float,int) or not math.isfinite(x) or not 0<x<=7200 for x in (warmup_s,duration_s)):raise ValueError('Invalid dynamic window')
    if phase=='formal' and (warmup_s<30 or duration_s<120):raise ValueError('Formal minimum is 30s warmup and 120s measurement')
    if type(change_tolerance_s) not in (float,int) or not math.isfinite(change_tolerance_s) or not 0<=change_tolerance_s<=600:raise ValueError('Explicit finite dispatch tolerance required')
    return dict(kind='dynamic_evaluation_protocol',phase=phase,warmup_s=warmup_s,duration_s=duration_s,
        warmup_min_success=warm,measure_min_success=measure_min_success,p99_min_success=1000,change_tolerance_s=change_tolerance_s,
        warmup='model calls on saved initial observations; no physical actions; fixed cutoff, no extension',
        admission='initially active class/component warmup; arrivals owe demand at committed admission with no excluded post-join warmup',
        terminal='hold_terminal: one physical episode per robot; no automatic reset; periodic obligations continue until departure',
        membership='join/leave commit times; cold replacement gaps remain in the fixed wall-clock denominator',
        sampling='at least measure_min_success per present class/component; p99 withheld below 1000 successful measured requests',
        failures='timeouts/rejections retained; unknown physical steps or failed cleanup invalidate the run',
        scope='project dynamic convention; not a steady-state capacity estimate or official paper equivalence')


def validate(p,spec,warmup_s,duration_s):
    if p!=protocol(spec,phase=p['phase'],warmup_s=warmup_s,duration_s=duration_s,**{k:p[k] for k in ('warmup_min_success','measure_min_success','change_tolerance_s')}):raise ValueError('Dynamic protocol changed')
    return p


def sample_counts(spec,robots,rows,*,start,end,minimum):
    classes={c['id']:c for c in spec['catalog']['fleet']['classes']};counts={c+'/'+m['id']:0 for c in set(robots.values()) for m in classes[c]['components']}
    for r in rows:
        if r['status']!='ok' or r['robot_id'] not in robots:continue
        if start<=r['invoked_ns']<end and r['returned_ns']<end:counts[robots[r['robot_id']]+'/'+r['component_id']]+=1
    short=sorted(k for k,n in counts.items() if n<minimum)
    return dict(start_ns=start,cutoff_ns=end,counts=counts,required=minimum,shortfall=short,requirements_met=not short)


def latency_summary(times):
    times=sorted(times);n=len(times)
    return dict(successes=n,p99_ms=times[math.ceil(.99*n)-1] if n>=1000 else None,p99_eligible=n>=1000)


def inspect_changes(events,journal,*,start,end,tolerance):
    dispatched=[r for r in journal if r['kind']=='scheduled_change_started'];completed=[r for r in journal if r['kind']=='scheduled_change_result']
    changes_ok=len(dispatched)==len(completed)==len(events);changes=[]
    failures=[r for r in journal if r['kind']=='health_failure' and r['time_ns']<end]
    recoveries=[]
    for f in failures:
        matches=[r for r in journal if r['kind']=='health_recovery_result' and r.get('worker_id')==f['worker_id'] and r.get('failed_epoch')==f['epoch'] and f['time_ns']<=r['time_ns']<=end]
        recovery=matches[0] if len(matches)==1 else None;result=recovery['result'] if recovery else {}
        committed=[r for r in journal if r['kind']=='population_committed' and r['epoch']==result.get('epoch') and f['time_ns']<=r['time_ns']<=recovery['time_ns']] if recovery else []
        ok=bool(recovery and result.get('status')=='committed' and result.get('epoch',0)>f['epoch'] and len(committed)==1)
        recoveries.append(dict(worker_id=f['worker_id'],failed_epoch=f['epoch'],failure_ns=f['time_ns'],recovery_ns=recovery['time_ns'] if recovery else None,accepted=ok,result=result))
    for i,e in enumerate(events):
        begun=dispatched[i] if i<len(dispatched) else None;done=completed[i] if i<len(completed) else None
        expected='injected' if e['op']=='fault' else 'committed'
        lag=(begun['time_ns']-start)/1e9-e['offset_s'] if begun else None
        ok=bool(begun and done and begun['index']==done['index']==i and begun['offset_s']==done['offset_s']==e['offset_s'] and
            0<=lag<=tolerance and begun['time_ns']<=done['time_ns']<=end and done['result']['status']==expected)
        if ok and e['op']=='fault':
            wid=done['result']['worker_id'];failed_epoch=done['result'].get('failed_epoch')
            injected=[r for r in journal if r['kind']=='owned_actor_fault_injected' and r['worker_id']==wid and r.get('epoch')==failed_epoch and begun['time_ns']<=r['time_ns']<=done['time_ns']]
            matched=[r for r in recoveries if r['worker_id']==wid and r['failed_epoch']==failed_epoch and r['failure_ns']>=begun['time_ns'] and r['accepted']]
            ok=len(injected)==len(matched)==1
        changes_ok=changes_ok and ok;changes.append(dict(index=i,op=e['op'],dispatch_lag_s=lag,accepted=ok,result=done['result'] if done else None))
    return changes,dict(requirements_met=changes_ok and all(r['accepted'] for r in recoveries),recoveries=recoveries)


def inspect_trial(root):
    import json
    from pathlib import Path
    from .profile_adoption import read
    from .dynamic_evidence import reaudit
    from .global_scenario import validate_evaluation
    p=Path(root);s=read(p/'scenario.json');validate_evaluation(s);policy=s['evaluation']
    if policy is None:raise ValueError('Diagnostic run has no prospective evaluation protocol')
    checked=reaudit(p);evaluation=checked['dynamic_evaluation'];a=evaluation['audit'];start,end=a['window_start_ns'],a['window_end_ns']
    if end-start!=round(s['duration_s']*1e9):raise ValueError('Dynamic duration changed')
    condition=read(p/'trial_condition.json')
    if condition!={'scenario_id':s['id'],'rep':condition['rep']} or type(condition['rep']) is not int or not 0<=condition['rep']<s['repetitions']:raise ValueError('Dynamic repetition identity differs')
    if s['spec']!=read(p/'control/spec.json'):raise ValueError('Scenario/control binding differs')
    expected_states={}
    for i,(rid,cid) in enumerate(s['robots'].items()):
        cases=s['banks'][cid]['manifest']['cases'];case=cases[(i+condition['rep'])%len(cases)]
        expected_states[rid]=dict(class_id=cid,seed=case['seed'],state_sha256=case['state_sha256'])
    if read(p/'simulation/initial_states.json')!=expected_states:raise ValueError('Prospective robot state/seed assignment differs')
    rows=[json.loads(x) for x in (p/'simulation/request_observations.jsonl').read_text().splitlines()]
    initial={r:c for c,rs in s['spec']['initial_population'].items() for r in rs}
    gate=read(p/'simulation/warmup_gate.json');want=sample_counts(s['spec'],initial,rows,start=gate['start_ns'],end=start,minimum=policy['warmup_min_success'])
    if gate['start_ns']!=start-round(s['warmup_s']*1e9):raise ValueError('Warmup origin changed')
    if gate!={**want,'decision_ns':gate['decision_ns']} or gate['decision_ns']<start:raise ValueError('Dynamic warmup gate changed')
    # Actions may only use a post-cutoff request, even if a warmup response arrived late.
    requests={r['request_id']:r for r in a['records'] if r['record_type']=='request'}
    if any(r['start_ns']<start or requests[r['request_id']]['sent_ns']<start for r in a['records'] if r['record_type']=='action'):raise ValueError('Physical action used warmup request')
    if not gate['requirements_met'] and any(r['sent_ns']>=start for r in requests.values()):raise ValueError('New request after rejected warmup')
    sample=sample_counts(s['spec'],s['robots'],rows,start=start,end=end,minimum=policy['measure_min_success'])
    journal=[json.loads(x) for x in (p/'control/journal.jsonl').read_text().splitlines()]
    changes,change_gate=inspect_changes(s['changes'],journal,start=start,end=end,tolerance=policy['change_tolerance_s'])
    statuses=read(p/'simulation/result.json');quality=gate['requirements_met'] and sample['requirements_met'] and change_gate['requirements_met'] and checked['status']=='complete' and statuses['status']=='complete'
    latency={}
    byid={r['request_id'].split(':',1)[1]:r for r in requests.values()}
    for cid,components in (evaluation['metrics'] or {}).get('component_request_metrics',{}).items():
        latency[cid]={}
        for mid,m in components.items():
            times=sorted((byid[r['request_id'].split(':',1)[1]]['finished_ns']-byid[r['request_id'].split(':',1)[1]]['sent_ns'])/1e6 for r in rows if r['status']=='ok' and s['robots'][r['robot_id']]==cid and r['component_id']==mid and start<=r['invoked_ns']<end and r['returned_ns']<end)
            latency[cid][mid]=latency_summary(times)
    return dict(kind='dynamic_trial_acceptance',status='accepted' if quality else 'not_accepted',phase=policy['phase'],run_id=a['run_id'],rep=condition['rep'],
        scenario_id=s['id'],warmup_gate=gate,sample_gate=sample,changes=changes,change_gate=change_gate,weighted_Q=evaluation['metrics']['weighted_Q'] if quality else None,
        raw_weighted_Q=evaluation['metrics']['weighted_Q'] if evaluation['metrics'] else None,latency=latency,
        outcomes=statuses['robot_outcomes'],formal_protocol_met=quality and policy['phase']=='formal',paper_equivalence_claim=False)


def report(scenario,root,directories):
    from pathlib import Path
    from .profile_adoption import inside,read
    from .global_scenario import validate_evaluation
    from .comparison_report import stats
    validate_evaluation(scenario)
    if scenario['evaluation'] is None or set(directories)!={str(i) for i in range(scenario['repetitions'])} or len(set(directories.values()))!=len(directories):raise ValueError('All distinct prospective repetitions required')
    rows=[];seen=set()
    for rep,path in sorted(directories.items(),key=lambda x:int(x[0])):
        folder=inside(root,path)
        if not folder.exists():rows.append(dict(rep=int(rep),directory=path,status='missing'));continue
        try:
            if read(folder/'scenario.json')!=scenario:raise ValueError('Dynamic scenario differs between repetitions')
            r=inspect_trial(folder)
            if r['rep']!=int(rep) or r['run_id'] in seen:raise ValueError('Repeated or misbound trial')
            seen.add(r['run_id']);rows.append(dict(r,directory=path))
        except (ValueError,KeyError,TypeError,OSError) as exc:rows.append(dict(rep=int(rep),directory=path,status='invalid',reason=str(exc)))
    complete=all(r['status']=='accepted' for r in rows)
    return dict(kind='dynamic_repeated_comparison',scenario_id=scenario['id'],rows=rows,status='complete' if complete else 'incomplete',
        weighted_Q=stats([r['weighted_Q'] for r in rows]) if complete else None,
        uncertainty_unit='prospectively repeated trial, not requests or frames; same tasks may repeat',
        phase=scenario['evaluation']['phase'],paper_equivalence_claim=False)
