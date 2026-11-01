"""Audit action-decision stop rates without conflating them with task success."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'src').is_dir():sys.path.insert(0,str(ROOT/'src'))


def assess_class(metrics, *, target=.9, minimum=100):
    n=metrics['decisions']; stopped=metrics['safety_stopped_decisions']; applied=metrics['raw_action_chunks']
    if any(type(x) is not int or x<0 for x in (n,stopped,applied)) or stopped+applied>n:
        raise ValueError('Inconsistent action-decision counts')
    rate=stopped/n if n else None
    return dict(decisions=n,safety_stopped_decisions=stopped,applied_action_chunks=applied,
        stop_rate=rate,target_maximum_stop_rate=target,minimum_decisions=minimum,
        target_met=n>=minimum and rate<=target and applied>0,
        denominator='audited non-warmup action decisions inside the measurement window',
        task_success_rate=False,semantic_quality_certified=False)


def report(raw):
    from rosa_repro.closed_loop_runtime import reaudit_closed
    raw=Path(raw); read=lambda name:json.loads((raw/name).read_text())
    rows=lambda name:[json.loads(s) for s in (raw/name).read_text().splitlines()]
    audit=reaudit_closed(raw); window=read('audit.json'); plan=read('execution_plan.json')
    classes={r['robot_id']:c['audit']['id'] for c in plan['classes'] for r in c['robots']}
    requests=[r for r in rows('requests.jsonl') if not r['warmup'] and window['window_start_ns']<=r['scheduled_ns']<window['window_end_ns']]
    details={r['request_id']:r for r in rows('request_details.jsonl')}
    result=dict(raw_reaudit='passed',source_manifest=read('source_manifest.json'),
        plan_id=plan['plan_id'],runtime_result=read('result.json'),classes={},latency={},
        semantic_quality_certified=False,formal_performance_claim=False)
    for cid in ('P3','P4'):
        m=audit['metrics']['classes'][cid]; entry=assess_class(m)
        cohort=[r for r in requests if classes[r['robot_id']]==cid and r['component_id']=='safety']
        labels=Counter(); images=set()
        for row in cohort:
            evidence=details[row['request_id']].get('backend_evidence') or {}
            if evidence.get('parse_status')=='ok':
                value=json.loads(evidence['raw_generation']);labels[value['decision'] if 'decision' in value else json.dumps(value,sort_keys=True)]+=1
            else:labels['invalid_or_missing']+=1
            images.add(tuple(i['sha256'] for i in evidence.get('images',[])))
        episodes=[e for e in rows('episodes.jsonl') if e['class_id']==cid]
        measured_episodes=[e for e in episodes if window['window_start_ns']<=e['start_ns']<window['window_end_ns']]
        entry.update(safety_request_labels=dict(labels),unique_safety_image_pairs=len(images),
            episodes_started_in_window=len(measured_episodes),
            episode_terminations=dict(Counter(e['termination'] for e in measured_episodes)),
            initial_state_ids_started_in_window=sorted({e['case_id'] for e in measured_episodes}),
            all_run_actual_simulator_steps=sum(e['actual_steps'] for e in episodes),
            qualified_action_chunks=m['qualified_action_chunks'])
        if plan.get('collision_guard'):
            lifecycle=rows('lifecycle.jsonl')
            gates=[x for x in lifecycle if x['kind']=='hybrid_safety_gate' and classes[x['robot_id']]==cid and window['window_start_ns']<=x['checked_at_ns']<window['window_end_ns']]
            previews=[x['evidence'] for x in lifecycle if x['kind']=='collision_guard_preview' and classes[x['robot_id']]==cid and window['window_start_ns']<=x['start_ns']<window['window_end_ns']]
            times=sorted(x['elapsed_ms'] for x in previews)
            entry['hybrid_guard']=dict(decision_causes=dict(Counter(x['cause'] for x in gates)),preview_statuses=dict(Counter(x['status'] for x in previews)),
                preview_p99_ms=times[math.ceil(.99*len(times))-1] if times else None,preview_maximum_ms=max(times) if times else None,
                preview_budget_ms=plan['collision_guard']['max_preview_ms'],separate_from_periodic_VLM_deadline=True,
                physics_preview_maximum_ms=max((x['physics_preview_ms'] for x in previews),default=None),
                recovery_maximum_ms=max((x.get('recovery_ms',0) for x in previews),default=None),
                violations=dict(Counter(json.dumps(x['violation'],sort_keys=True) for x in previews if 'violation' in x)))
        result['classes'][cid]=entry
    for c in plan['classes']:
        cid=c['audit']['id']
        for component in c['audit']['components']:
            key=cid+'/'+component['id'];cohort=[r for r in requests if classes[r['robot_id']]==cid and r['component_id']==component['id']]
            ok=[r for r in cohort if r['status']=='ok' and r['valid_output']]
            xs=sorted((r['finished_ns']-r['scheduled_ns'])/1e6 for r in ok)
            p99=xs[math.ceil(.99*len(xs))-1] if xs else None
            slots=audit['metrics']['periodic_slot_metrics'][cid].get(component['id'])
            result['latency'][key]=dict(n=len(cohort),valid=len(ok),p99_ms=p99,maximum_ms=max(xs) if xs else None,
                deadline_ms=component['deadline_ms'],periodic_slots=slots,
                target_met=bool(xs) and len(ok)==len(cohort) and p99<=component['deadline_ms'] and (slots is None or slots['slot_meet_rate']>=.99))
    result['stop_target_met']=result['runtime_result']['status']=='complete' and all(v['target_met'] for v in result['classes'].values())
    result['latency_targets_met']=all(v['target_met'] for v in result['latency'].values())
    if audit.get('recovery'):
        result['recovery']=audit['recovery']
        result['recovery_target_met']=result['stop_target_met'] and all(
            v['final_stop_rate'] is not None and v['final_stop_rate']<=.9 and v['retreats_applied']>0 and v['sustained_s1_resumptions']>0
            for v in audit['recovery']['classes'].values())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();value=report(args.raw)
    with args.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value,indent=2))
