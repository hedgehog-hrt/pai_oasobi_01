"""Recomputed fixed-state task evidence required for S1 profile admission."""
from .execution_plan import runtime_source_manifest
from .r06_protocol import validate,TASKS


def assess(report,selected_batch):
    if report['status']!='complete' or report['mode']!='libero_gpu':raise ValueError('R06 requires complete formal GPU task trials')
    for row in report['comparisons']:
        if not row['statistics'] or not row['statistics']['all_pairs_valid']:raise ValueError('R06 paired task/state evidence incomplete')
    coverage={}
    for task in TASKS:
        rows=[r for r in report['rows'] if r['cell']['task']==task and r['cell']['arm']=='selected']
        seen=sorted({b for r in rows for b in r['audit']['batch_sizes']})
        if selected_batch not in seen:raise ValueError('R06 selected batch never observed for '+task)
        coverage[task]=seen
    return dict(report=report,actual_selected_batch_coverage=coverage,selected_batch=selected_batch,
        automatic_quality_equivalence_claim=False,
        unresolved_comparisons=[r['candidate'] for r in report['comparisons'] if r['statistics']['decision']=='unresolved'],
        investigate_drop_comparisons=[r['candidate'] for r in report['comparisons'] if r['statistics']['decision']=='investigate_drop'])


def quality(e,q,c,server):
    from .statistical_evaluation import r06_report
    spec=q['r06']
    if set(spec)!={'suite','bank_root','directories'}:raise ValueError('R06 suite, bank and prospective directory map required')
    suite=e.json(spec['suite']);bank=e.tree(spec['bank_root']);mapping=e.json(spec['directories'])
    validate(suite,bank)
    if suite['mode']!='libero_gpu' or suite['phase']!='formal' or suite['source_manifest']!=runtime_source_manifest():raise ValueError('R06 formal GPU/current source required')
    if suite['server_evidence']!=server:raise ValueError('R06 checkpoint/runtime differs from numerical regression')
    if any(cell['max_batch_size']!=c['batch_size'] for cell in suite['cells'] if cell['arm']=='selected'):raise ValueError('R06 candidate batch differs from profile')
    for path in mapping.values():e.tree(path)
    report=r06_report(suite,e.root,mapping)
    return assess(report,c['batch_size'])


def review_checks(quality):
    return {r['candidate']:dict(decision=r['statistics']['decision'],success_difference=r['statistics']['success_difference'],
        confidence_interval=r['statistics']['confidence_interval'],accepted=None,reason=None)
        for r in quality['report']['comparisons']}


def validate_review(quality,review):
    expected=review_checks(quality)
    if set(review)!=set(expected):raise ValueError('Explicit review of every R06 task/width/arm comparison required')
    for key,entry in expected.items():
        row=review[key]
        if set(row)!=set(entry) or any(row[k]!=entry[k] for k in ('decision','success_difference','confidence_interval')):raise ValueError('R06 review statistics changed')
        if row['accepted'] is not True or not isinstance(row['reason'],str) or not row['reason'].strip():raise ValueError('R06 comparison review incomplete or rejected')
