"""Review measured joint residency; samples are not claimed to be allocation peaks."""
import json,math
from pathlib import Path
from .dedicated_execution import validate
from .execution_plan import digest,runtime_source_manifest
from .profile_adoption import inside,read,runtime_signature
from .regression import sha256


def inspect(plan,root,directories):
    from .closed_loop_runtime import reaudit_closed
    validate(plan)
    if plan['mode']!='gpu':raise ValueError('CPU runs cannot establish GPU joint residency')
    if not isinstance(directories,list) or not 3<=len(directories)<=10 or len(set(directories))!=len(directories):raise ValueError('Three to ten distinct prospective GPU repetitions required')
    rows=[];ids=set();reps=set();suite_id=None;signatures=None
    for name in directories:
        p=inside(root,name);checked=reaudit_closed(p);r=read(p/'result.json');s=read(p/'suite.json')
        launch=read(p/'dedicated_launch.json')
        if read(p/'execution_plan.json')!=plan or launch['mode']!='GPU_joint_residency_calibration' or launch['comparison_eligible']:raise ValueError('Dedicated calibration binding differs')
        if r['status']!='complete' or not r['formal_protocol_met'] or not r['drained']:raise ValueError('Incomplete or insufficient residency measurement')
        if s['protocol']['phase']!='formal' or not checked['measurement_started'] or not checked['warmup_gate']['requirements_met'] or not checked['samples']['requirements_met']:raise ValueError('Formal residency gates not met')
        if r['run_id'] in ids:raise ValueError('Repeated raw run')
        ids.add(r['run_id']);rep=read(p/'condition.json')['rep']
        if rep in reps or (suite_id is not None and suite_id!=s['suite_id']):raise ValueError('Repetitions must belong to the same prospective suite')
        reps.add(rep);suite_id=s['suite_id']
        before=read(p/'workers_before.json');after=read(p/'workers_after.json');env=read(p/'worker_environments.json')
        from .dedicated_execution import check_inventory
        check_inventory(plan,before);check_inventory(plan,after)
        signature={w:runtime_signature(env[w],before[w]['backend_attestation']) for w in before}
        if signatures is not None and signature!=signatures:raise ValueError('Runtime changed across residency trials')
        signatures=signature
        for w in before:
            if before[w]['instance_id']!=after[w]['instance_id'] or after[w].get('group_peak_invocations')!=1 or after[w]['inflight_batches'] or after[w]['busy_sessions']:raise ValueError('Residency changed, overlapped or did not drain')
            if runtime_signature(env[w],after[w]['backend_attestation'])!=signature[w]:raise ValueError('Backend changed during residency measurement')
        telemetry=[json.loads(x) for x in (p/'telemetry.jsonl').read_text().splitlines()];a=read(p/'audit.json');groups=[]
        if any(x['status']!='ok' for x in a['records'] if x['record_type']=='request'):raise ValueError('Failed requests in residency candidate')
        for g in plan['physical_groups']:
            # All logical workers may poll the same device. Aggregate by physical
            # group; never add their identical whole-device memory samples.
            samples=[x['sample'] for x in telemetry if x['worker_id'] in g['worker_ids'] and a['window_start_ns']<=x['driver_ns']<a['window_end_ns'] and 'sample' in x]
            uuid=before[g['worker_ids'][0]]['gpu_properties']['uuid']
            if len(samples)<2 or any(x.get('kind')!='nvidia_smi_sample' or x['uuid']!=uuid or
                any(type(x[k]) not in (float,int) or not math.isfinite(x[k]) for k in ('used_mib','total_mib')) or not 0<x['used_mib']<x['total_mib'] for x in samples):raise ValueError('Missing or invalid joint residency telemetry')
            groups.append(dict(group_id=g['group_id'],sample_count=len(samples),maximum_observed_device_mib=max(x['used_mib'] for x in samples),
                minimum_observed_headroom_mib=min(x['total_mib']-x['used_mib'] for x in samples)))
        rows.append(dict(directory=name,evidence_sha256=sha256(p/'evidence.json'),run_id=r['run_id'],rep=rep,groups=groups,
            request_metrics=checked['metrics']['component_request_metrics']))
    result=dict(kind='dedicated_joint_residency_candidate',plan_id=plan['plan_id'],source_manifest=runtime_source_manifest(),
        directories=list(directories),runtime_signatures=signatures,rows=rows,eligible_for_review=True,
        memory_semantics='sampled whole-device memory, not peak allocation; manual headroom review required',
        latency_semantics='measured serialized co-resident workload; no single-model latency substitution',paper_performance_claim=False)
    result['id']=digest(result);return result


def template(candidate):
    return dict(candidate_id=candidate['id'],approved=False,reviewer='',reason='',joint_memory_headroom_accepted=False,
        serialized_latency_and_task_metrics_accepted=False)


def validate_review(plan,bundle):
    if not isinstance(bundle,dict) or set(bundle)!={'evidence_root','candidate','review'}:raise ValueError('Reviewed joint residency bundle required')
    root=Path(bundle['evidence_root']);candidate=read(inside(root,bundle['candidate']));review=read(inside(root,bundle['review']))
    expected=inspect(plan,root,candidate['directories'])
    if candidate!=expected or set(review)!=set(template(candidate)) or review['candidate_id']!=candidate['id']:raise ValueError('Residency evidence/review differs')
    if any(review[k] is not True for k in ('approved','joint_memory_headroom_accepted','serialized_latency_and_task_metrics_accepted')) or any(not isinstance(review[k],str) or not review[k].strip() for k in ('reviewer','reason')):raise ValueError('Joint residency review is not approved')
    return candidate
