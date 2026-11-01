"""Integrity of global-control evidence; dynamic P1-P4 conversion is separate."""
import json
from pathlib import Path
from .execution_plan import digest
from .regression import sha256,write_json


def seal_tree(root):
    p=Path(root);files={str(f.relative_to(p)):sha256(f) for f in sorted(p.rglob('*')) if f.is_file() and f!=p/'tree-evidence.json'}
    write_json(p/'tree-evidence.json',dict(files=files,sha256=digest(files)))


def verify_tree(root):
    p=Path(root);e=json.loads((p/'tree-evidence.json').read_text())
    if digest(e['files'])!=e['sha256'] or set(e['files'])!={str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f!=p/'tree-evidence.json'}:raise ValueError('Global evidence inventory changed')
    for name,h in e['files'].items():
        f=(p/name).resolve()
        if not f.is_relative_to(p.resolve()) or (p/name).is_symlink() or sha256(f)!=h:raise ValueError('Global evidence modified')
    return e


def evaluate(root):
    """Recompute one fixed-window P1-P4 audit across all admitted generations.

    Gaps in model availability become empty episodes with missing periodic slots;
    they are not removed from the wall-clock denominator. Unknown physical action
    completion invalidates the trial rather than being converted to zero steps.
    """
    from .global_control import validate,check_decision
    from .dynamic_control import reaudit as audit_control
    from .multicomponent import evaluate_multicomponent,ROLES
    from .execution_plan import runtime_source_manifest
    from .component_runtime import validate_output
    from .paired import digest_value
    from .profile_adoption import inside
    import math
    p=Path(root);verify_tree(p);read=lambda f:json.loads((p/f).read_text());spec=validate(read('spec.json'))
    rows=[json.loads(line) for line in (p/'journal.jsonl').read_text().splitlines()]
    if not rows or rows[0]['kind']!='started' or rows[-1]['kind']!='closed' or rows[0]['spec_id']!=spec['id']:raise ValueError('Global lifecycle incomplete')
    if any(a['time_ns']>b['time_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('Global clock order differs')
    windows=[r for r in rows if r['kind']=='measurement_window']
    if len(windows)!=1:raise ValueError('Exactly one prospective measurement window required')
    window=windows[0];start,end=window['window_start_ns'],window['window_end_ns'];audit_end=rows[-1]['time_ns'];epoch=rows[0]['time_ns']
    if not epoch<=window['time_ns']<start<end or window['run_epoch_ns']!=epoch:raise ValueError('Window is post-selected')
    if end>audit_end and rows[-1]['clean']:raise ValueError('Complete trial ended before its fixed cutoff')
    known={c['id']:c for c in spec['catalog']['fleet']['classes']};classes=[]
    for cid,c in known.items():
        comps=[]
        for m in c['components']:
            role=spec['catalog']['roles'][m['id']];v=dict(id=m['id'],role=role,deadline_ms=m['deadline_ms'])
            if role in ('safety','monitor'):
                period=1e9/m['periodic_hz']
                if not math.isclose(period,round(period),abs_tol=1e-6):raise ValueError('Fractional periodic clock')
                v.update(period_ns=round(period),phase_ns=0)
            comps.append(v)
        pipeline=next(k for k,v in ROLES.items() if v=={m['role'] for m in comps})
        classes.append(dict(id=cid,pipeline=pipeline,weight=c['weight'],s2_horizon=c['s2_horizon'],components=comps))
    classmap={c['id']:c for c in classes};ready={};closed={};ownership=0;residence={};active={};attempts={};commits=[];epoch_specs={}
    for r in rows:
        k=r['kind']
        if k=='plan_requested':attempts[r['attempt']]=r
        if k=='generation_ready':
            if ownership:raise ValueError('Overlapping fleet generations')
            directory=r['directory'];folder=inside(p,directory);cs=json.loads((folder/'spec.json').read_text())
            if cs['blueprint']['plan_id']!=r['plan_id'] or len(cs['workers'])!=r['physical_leases'] or r['physical_leases']>cs['physical_gpu_budget']:raise ValueError('Generation resource/plan mismatch')
            ownership=r['physical_leases'];ready[directory]=r
            if r['epoch'] in epoch_specs:raise ValueError('Generation epoch reused')
            epoch_specs[r['epoch']]=cs
        if k=='generation_closed':closed[r['directory']]=r
        if k=='leases_released':
            if r['remaining_owned_leases']!=0:raise ValueError('Unreleased old leases')
            ownership=0
        if k in ('population_committed','rollback_committed'):
            wanted={rid:cid for cid,rs in r['population'].items() for rid in rs}
            if len(wanted)!=sum(map(len,r['population'].values())) or set(r['population'])!=set(known):raise ValueError('Membership identity mismatch')
            if wanted:
                cs=epoch_specs.get(r['epoch'])
                if cs is None or r['gpu_budget']!=cs['physical_gpu_budget'] or set(cs['initial_active'])!=set(wanted):raise ValueError('Committed population/budget differs from installed generation')
                deployed={x['robot_id']:c['audit']['id'] for c in cs['blueprint']['classes'] for x in c['robots']}
                if deployed!=wanted:raise ValueError('Committed robot classes differ from installed plan')
            elif ownership:raise ValueError('Idle population still owns model actors')
            for rid,(cid,begin) in list(active.items()):
                if wanted.get(rid)!=cid:residence.setdefault(rid,[]).append((cid,begin,r['time_ns']));del active[rid]
            for rid,cid in wanted.items():
                if rid not in active:active[rid]=(cid,r['time_ns'])
            commits.append(r)
    for rid,(cid,begin) in active.items():residence.setdefault(rid,[]).append((cid,begin,audit_end))
    if rows[-1]['clean'] and ownership:raise ValueError('Complete control trace retains model leases')
    for f in p.glob('decision-*.json'):
        d=json.loads(f.read_text());req=attempts[d['revision']];check_decision(spec,d,req['population'],req['gpu_budget'],d['revision'])
    generations=read('control_result.json')['generations']
    if len(generations)!=len(set(generations)) or set(ready)-set(generations) or set(closed)-set(generations):raise ValueError('Generation inventory differs')
    records=[];episodes=[];coverage={};unknown=[];details=[]
    run=rows[0]['run_id'];domain=run+'-driver-monotonic'
    for directory in generations:
        folder=inside(p,directory);result=audit_control(folder);events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
        if result['unknown_actions']:unknown.append(directory)
        if directory not in ready:
            if result['predictions'] or result['actions_started']:raise ValueError('Uncommitted generation executed work')
            continue
        if directory not in closed:raise ValueError('Generation closure missing')
        if not events[0]['time_ns']<=ready[directory]['time_ns']<=closed[directory]['time_ns']<=audit_end:raise ValueError('Generation lifecycle clock differs')
        cp=json.loads((folder/'spec.json').read_text())['blueprint'];robot_class={r['robot_id']:c['audit']['id'] for c in cp['classes'] for r in c['robots']}
        for c in cp['classes']:
            if c['audit']!=classmap[c['audit']['id']]:raise ValueError('Class semantics changed during trial')
        starts={};ends={};resets={};action_starts={};planned_end=events[-1]['time_ns'];epmap={}
        for e in events:
            if e['kind']=='reset':
                rid=e['robot_id'];eid=e['episode_id'];cid=robot_class[rid]
                spans=[(max(e['time_ns'],b),min(planned_end,d)) for cl,b,d in residence.get(rid,[]) if cl==cid and b<planned_end and d>e['time_ns']]
                if len(spans)>1:raise ValueError('Ambiguous generation membership')
                if spans:
                    b,d=spans[0];ep=dict(class_id=cid,robot_id=rid,episode_id=directory+':'+eid,start_ns=b,end_ns=d)
                    if b<d:episodes.append(ep);coverage.setdefault(rid,[]).append(ep);epmap[eid]=ep
                resets[rid]=e
            elif e['kind']=='prediction_started':starts[e['metadata']['request_id']]=e
            elif e['kind'] in ('prediction_finished','prediction_failed'):ends[e['request_id']]=e
            elif e['kind']=='action_started':action_starts[e['ticket_id']]=e
        for qid,s in starts.items():
            e=ends[qid];meta=s['metadata'];rid=s['robot_id'];ep=epmap.get(meta['episode_id'])
            if not ep:raise ValueError('Request outside admitted membership')
            model=next(m for m in classmap[ep['class_id']]['components'] if m['id']==meta['component_id'])
            if e['kind']=='prediction_failed':status='timeout' if e['error_type']=='TimeoutError' else 'error'
            else:status=e['status'] if e['status'] in ('ok','error','timeout','rejected','cancelled') else 'rejected'
            semantic=e.get('semantic_output')
            if status=='ok' and model['role']!='s1':
                validate_output(model['role'],semantic)
                if digest_value(semantic)!=e['output_sha256']:raise ValueError('Semantic output changed')
            scheduled=s.get('scheduled_ns');slot=s.get('slot_index')
            if model['role'] in ('monitor','safety') and (scheduled is None or slot is None):raise ValueError('Periodic request lacks global slot')
            pid=s.get('plan_request_id')
            record=dict(schema_version='2.0',record_type='request',run_id=run,clock_domain=domain,robot_id=rid,episode_id=ep['episode_id'],request_id=directory+':'+qid,
                attempt_id=meta['attempt_id'],component_id=meta['component_id'],observation_seq=meta['observation_seq'],scheduled_ns=s['time_ns'] if scheduled is None else scheduled,
                sent_ns=s['time_ns'],finished_ns=e['time_ns'],deadline_ms=s['deadline_ms'],status=status,valid_output=status=='ok',warmup=s['time_ns']<start,
                batch_id=e.get('batch_id'),actual_batch_size=e.get('actual_batch_size'),slot_index=slot,plan_request_id=directory+':'+pid if pid else None)
            records.append(record)
        for e in events:
            if e['kind'] not in ('action_finished','action_stopped'):continue
            if e.get('status')=='unknown':continue  # Entire trial is invalidated below.
            qid=e['ticket_id'];s=starts[qid];meta=s['metadata'];ep=epmap[meta['episode_id']];stopped=e['kind']=='action_stopped';begin=e if stopped else action_starts[qid]
            records.append(dict(schema_version='2.0',record_type='action',run_id=run,clock_domain=domain,robot_id=e['robot_id'],episode_id=ep['episode_id'],request_id=directory+':'+qid,
                component_id=meta['component_id'],action_id=directory+':action:'+qid,start_ns=begin['time_ns'],end_ns=e['time_ns'],applied=False if stopped else e['steps']>0,
                executed_steps=0 if stopped else e['steps'],safety_stopped=stopped))
        details.append(dict(directory=directory,control=result))
    # Explicit demand intervals with no available model generation still owe
    # periodic slots. They are not silently removed during cold migration.
    gaps=[]
    for rid,spans in residence.items():
        for cid,begin,finish in spans:
            cursor=begin
            for ep in sorted(coverage.get(rid,[]),key=lambda e:e['start_ns']):
                if ep['end_ns']<=begin or ep['start_ns']>=finish:continue
                if ep['start_ns']<cursor:raise ValueError('Overlapping generation episodes')
                if cursor<ep['start_ns']:gaps.append(dict(class_id=cid,robot_id=rid,episode_id='downtime-'+str(len(gaps)),start_ns=cursor,end_ns=ep['start_ns']))
                cursor=ep['end_ns']
            if cursor<finish:gaps.append(dict(class_id=cid,robot_id=rid,episode_id='downtime-'+str(len(gaps)),start_ns=cursor,end_ns=finish))
    episodes+=gaps;data=dict(schema_version='2.0',run_id=run,clock_domain=domain,run_epoch_ns=epoch,window_start_ns=start,window_end_ns=end,audit_end_ns=audit_end,classes=classes,episodes=episodes,records=records)
    incomplete=unknown or not rows[-1]['clean'] or read('control_result.json')['status']!='complete'
    metrics=None if incomplete else evaluate_multicomponent(data)
    return dict(kind='dynamic_pipeline_evaluation',status='incomplete' if incomplete else 'complete',metrics=metrics,audit=data,unknown_action_generations=unknown,
        downtime_episodes=gaps,control_generations=details,source_manifest=runtime_source_manifest(),mode=spec['catalog']['mode'],
        periodic_contract='global clock across generations; admitted membership owes slots during model downtime',
        formal_GPU_performance_claim=False,paper_metric_equivalence_claim=False,robot_safety_certified=False)
