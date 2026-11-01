"""Prospective-cell reports from raw P1-P4/R07 evidence. No paper values invented."""
from collections import Counter
from pathlib import Path
import json,statistics
from .execution_plan import digest,validate_execution_plan
from .profile_adoption import read,inside
from .regression import sha256
from .multicomponent import evaluate_multicomponent
from .heterogeneous import Fleet

EXPECTED={'execution_mode','action_semantics','source_sha256','workload_sha256','protocol_sha256'}


def condition(directory):
    p=Path(directory);plan=validate_execution_plan(read(p/'execution_plan.json'));protocol=read(p/'protocol.json');source=read(p/'source_manifest.json')
    if source['sha256']!=digest(source['files']):raise ValueError('Source manifest digest mismatch')
    return dict(execution_mode=plan['mode'],action_semantics=protocol['action_semantics'],source_sha256=source['sha256'],
        workload_sha256=digest(dict(classes=plan['inputs']['fleet']['classes'],hardware=plan['inputs']['fleet']['hardware'])),
        protocol_sha256=digest(protocol))


def audit_pipeline(directory):
    p=Path(directory);c=condition(p);plan=read(p/'execution_plan.json');result=read(p/'result.json')
    closed=None
    if result.get('trial_kind')=='closed_loop' and (p/'audit.json').exists():
        from .closed_loop_runtime import reaudit_closed
        closed=reaudit_closed(p)
    if result['plan_id']!=plan['plan_id'] or result['execution_mode']!=c['execution_mode'] or result['action_semantics']!=c['action_semantics']:
        raise ValueError('Result identity mismatch')
    if plan.get('purpose')=='dedicated_GPU_execution' and read(p/'dedicated_launch.json')['mode']=='GPU_joint_residency_calibration':
        return dict(status='failed',reason='joint_residency_calibration_not_comparative_run',condition=c,closed_loop_audit=closed)
    if result['status'] not in ('complete','complete_with_request_failures') or result.get('drained') is not True:
        return dict(status='failed',reason='run_incomplete_or_not_drained',condition=c,closed_loop_audit=closed)
    a=read(p/'audit.json');rows=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
    if a['records']!=rows or a['run_id']!=result['run_id']:raise ValueError('Raw event/audit identity mismatch')
    if a['classes']!=[x['audit'] for x in plan['classes']]:raise ValueError('Audit workload differs from execution plan')
    m=evaluate_multicomponent(a)
    if m!=read(p/'metrics.json') or any(m[k]!=result[k] for k in ('weighted_Q','qualified_action_chunks')):
        raise ValueError('Recomputed metrics disagree with saved summaries')
    requests=[r for r in rows if r['record_type']=='request']
    if Counter(r['status'] for r in requests)!=result['request_statuses'] or len(requests)!=result['requests']:
        raise ValueError('Raw request counts differ')
    status='complete' if all(r['status']=='ok' for r in requests) else 'complete_with_request_failures'
    if result['status']!=status:raise ValueError('Result status disagrees with raw failures')
    fleet=Fleet.from_dict(plan['inputs']['fleet']);predicted=None if plan.get('purpose') in ('paper_fixed_allocation_comparison','dedicated_GPU_execution') else fleet.objective(plan['inputs']['schedule']['frequencies_hz'])
    return dict(status='passed' if status=='complete' else 'failed',reason=None if status=='complete' else 'request_failures',
        condition=c,run_id=result['run_id'],observed_weighted_Q=m['weighted_Q'],predicted_weighted_objective=predicted,
        qualified_action_chunks=m['qualified_action_chunks'],duration_s=m['duration_s'],classes=m['classes'],
        request_metrics=m['component_request_metrics'],periodic_slot_metrics=m['periodic_slot_metrics'],
        formal_performance_claim=result.get('formal_performance_claim',False),
        closed_loop_audit=closed,
        paper_official_metric_equivalence_verified=False,
        evidence_files={f.name:sha256(f) for f in sorted(p.iterdir()) if f.is_file()})


def validate_manifest(m):
    if set(m)!={'schema_version','purpose','conditions','cells'} or m['schema_version']!='1.0' or m['purpose']!='prospective_pipeline_comparison':
        raise ValueError('Invalid comparison manifest')
    if not m['conditions'] or not m['cells']:raise ValueError('Expected conditions and cells required')
    for name,value in m['conditions'].items():
        if not isinstance(name,str) or not name or set(value)!=EXPECTED:raise ValueError('Explicit mode/source/workload/protocol conditions required')
        if value['execution_mode'] not in ('gpu','cpu_dummy') or value['action_semantics'] not in ('virtual_timed_chunks','external_executor_reported_steps'):
            raise ValueError('Unknown provenance')
        for key in ('source_sha256','workload_sha256','protocol_sha256'):
            if len(value[key])!=64 or any(c not in '0123456789abcdef' for c in value[key]):raise ValueError('Invalid condition hash')
    ids=set();cohorts=set();paths=set()
    for cell in m['cells']:
        if set(cell)!={'id','condition_id','variant','repetition','directory'} or not all(isinstance(cell[k],str) and cell[k] for k in ('id','variant','directory')):
            raise ValueError('Invalid expected cell')
        key=(cell['condition_id'],cell['variant'],cell['repetition'])
        if cell['condition_id'] not in m['conditions'] or type(cell['repetition']) is not int or cell['repetition']<0:raise ValueError('Invalid cell condition/repetition')
        if cell['id'] in ids or key in cohorts or cell['directory'] in paths:raise ValueError('Duplicate cell/repetition/evidence directory')
        ids.add(cell['id']);cohorts.add(key);paths.add(cell['directory'])
    return m


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
        sample_stddev=statistics.stdev(values) if len(values)>1 else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None,
        uncertainty_unit='independent declared run repetitions; no frame/request-level confidence intervals')


def report(manifest,root):
    validate_manifest(manifest);rows=[];identities=set()
    for cell in manifest['cells']:
        row=dict(cell)
        try:
            path=inside(root,cell['directory'])
            if not path.exists():row.update(status='missing',reason='expected_directory_absent')
            else:
                observed=audit_pipeline(path)
                if observed['condition']!=manifest['conditions'][cell['condition_id']]:raise ValueError('Execution conditions differ from declared comparison cell')
                if observed.get('run_id'):
                    if observed['run_id'] in identities:raise ValueError('Same run reused in more than one cell')
                    identities.add(observed['run_id'])
                row.update(observed)
        except (ValueError,KeyError,TypeError,OSError) as exc:row.update(status='invalid',reason=str(exc))
        rows.append(row)
    groups=[]
    for cid,variant in dict.fromkeys((x['condition_id'],x['variant']) for x in rows):
        selected=[r for r in rows if r['condition_id']==cid and r['variant']==variant];ok=[r for r in selected if r['status']=='passed']
        group=dict(condition_id=cid,variant=variant,expected_repetitions=len(selected),status_counts=dict(Counter(r['status'] for r in selected)),
            complete=len(ok)==len(selected),observed_weighted_Q=stats([r['observed_weighted_Q'] for r in ok]),
            predicted_weighted_objective=stats([r['predicted_weighted_objective'] for r in ok if r['predicted_weighted_objective'] is not None]),classes={})
        for klass in dict.fromkeys(k for r in ok for k in r['classes']):
            values=[r['classes'][klass] for r in ok]
            group['classes'][klass]=dict(pipeline=values[0]['pipeline'],weight=values[0]['weight'],
                qualified_action_rate_hz=stats([v['qualified_action_rate_hz'] for v in values]),
                qualified_action_chunks=stats([v['qualified_action_chunks'] for v in values]))
        groups.append(group)
    return dict(schema_version='1.0',manifest_sha256=digest(manifest),status='complete' if all(x['status']=='passed' for x in rows) else 'incomplete',
        rows=rows,groups=groups,conditions=manifest['conditions'],paper_reproduction_confirmed=False,
        limitations=['Means use successful repetitions only; counts and failed rows are retained. Incomplete groups are not comparative claims.',
          'Predicted R07 objective and observed qualified action rate are distinct quantities.',
          'Virtual replay, CPU dummy and connection-validation runs are not paper or robot task-success results.',
          'No paper baseline, speedup or confidence interval is inferred. Declare expected cells before collecting results.'])


def markdown(r):
    def safe(x):return str(x).replace('|','\\|').replace('\n',' ')
    def f(x):return '—' if x is None else f'{x:.6g}'
    lines=['# P1–P4 / R07 comparison report','',f"Status: **{r['status']}**. Paper reproduction confirmed: **false**.",'',
        '| Condition | Variant | Passed / expected | Missing / failed / invalid | Observed Q mean ± sample SD | Predicted objective mean |',
        '| --- | --- | ---: | --- | ---: | ---: |']
    for g in r['groups']:
        s=g['status_counts'];v=g['observed_weighted_Q']
        lines.append('| '+' | '.join(map(safe,[g['condition_id'],g['variant'],f"{s.get('passed',0)} / {g['expected_repetitions']}",
            f"{s.get('missing',0)} / {s.get('failed',0)} / {s.get('invalid',0)}",f"{f(v['mean'])} ± {f(v['sample_stddev'])}",f(g['predicted_weighted_objective']['mean'])]))+' |')
    lines+=['','| Condition / variant | Class | Pipeline | Qualified action rate mean ± sample SD (Hz) |','| --- | --- | --- | ---: |']
    for g in r['groups']:
        for klass,v in g['classes'].items():
            s=v['qualified_action_rate_hz'];lines.append(f"| {safe(g['condition_id'])} / {safe(g['variant'])} | {safe(klass)} | {v['pipeline']} | {f(s['mean'])} ± {f(s['sample_stddev'])} |")
    lines+=['','| Cell | Status | Reason |','| --- | --- | --- |']
    lines += ['| '+' | '.join(safe(x) for x in (row['id'],row['status'],row.get('reason') or ''))+' |' for row in r['rows']]
    lines+=['','Conditions:','', '```json',json.dumps(r['conditions'],indent=2),'```','']
    lines+=['- '+x for x in r['limitations']]
    return '\n'.join(lines)+'\n'
