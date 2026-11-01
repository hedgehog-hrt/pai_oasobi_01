"""Prospective R08/R09 comparison with complete repeated runs and raw re-audit."""
import json,statistics
from pathlib import Path
from .paper_conditions import validate
from .paper_baselines import PURPOSE
from .execution_plan import digest
from .profile_adoption import inside
from .comparison_report import audit_pipeline,stats


def family(cell):return ':'.join((cell['experiment'],cell['workload'],str(sum(c['robots'] for c in cell['classes']))))
def variant(cell):return 'ROSA' if cell['comparison']=='ROSA' else cell['comparison']+':'+cell['method']


def report(paper,root,directories,references):
    validate(paper);allcells={c['id']:c for c in paper['cells'] if c['experiment'] in ('R08','R09')}
    if not directories or set(directories)-set(allcells) or len(set(directories.values()))!=len(directories):raise ValueError('Distinct prospective paper cell paths required')
    families={family(allcells[cid]) for cid in directories}
    if set(directories)!={cid for cid,c in allcells.items() if family(c) in families} or set(references)!=families:raise ValueError('Every variant/repetition in each requested family is required')
    rows=[];runids=set();bindings={};batch_one_profiles={}
    for cid,path in directories.items():
        cell=allcells[cid];key=family(cell);row=dict(cell=cell,directory=path,family=key,variant=variant(cell));p=inside(root,path)
        if cell['comparison']=='dedicated_baseline' and cell['allocation']['shortfall']:
            row.update(status='resource_infeasible',reason='entire_workload_exceeds_dedicated_budget',shortfall=cell['allocation']['shortfall'],
                declared_GPU_budget=cell['dedicated_gpu_budget'],required_physical_gpus=cell['required_physical_gpus'])
        elif not p.exists():row.update(status='missing',reason='expected_directory_absent')
        else:
            try:
                observed=audit_pipeline(p);plan=json.loads((p/'execution_plan.json').read_text());result=json.loads((p/'result.json').read_text())
                reference=json.loads(inside(root,references[key]).read_text())
                from .execution_plan import validate_execution_plan
                validate_execution_plan(reference)
                from .paper_baselines import _workload
                _workload(reference,cell)
                if cell['comparison']=='ROSA':
                    if plan!=reference:raise ValueError('ROSA run differs from reference')
                elif cell['comparison']=='dedicated_baseline':
                    from .dedicated_comparison import compile_cell
                    if plan.get('purpose')!='dedicated_GPU_execution' or plan!=compile_cell(reference,paper,cid,plan['inputs']['resource_plan']['profiles']):raise ValueError('Dedicated plan differs from paper/reference/budget')
                elif plan.get('purpose')!=PURPOSE or plan['comparison']['cell']!=cell or plan['inputs']['reference']!=reference or plan['inputs']['paper_inputs']!=paper['inputs']:
                    raise ValueError('Baseline run differs from prospective paper/reference')
                if result.get('trial_kind')!='closed_loop' or json.loads((p/'condition.json').read_text()).get('rep')!=cell['repeat']:raise ValueError('Expected matching live trial repetition')
                if observed['status']!='passed':raise ValueError('Failed/incomplete trial retained without comparative speedup')
                run=observed['run_id']
                if run in runids:raise ValueError('Reused run in comparison')
                runids.add(run)
                suite=json.loads((p/'suite.json').read_text());pro=suite['protocol']
                controls={k:v for k,v in pro.items() if k not in ('nominal_rates_hz','recommended_timing')}
                bound=dict(controls=controls,banks=suite['banks'],source=suite['source_manifest'],mode=plan['mode'],
                    model_bindings=reference['inputs']['bindings'])
                if key in bindings and bindings[key]!=bound:raise ValueError('Input/runtime/protocol conditions differ across variants')
                if cell['comparison'] in ('full_baseline','dedicated_baseline'):
                    b1=plan['inputs']['resource_plan']['profiles'] if cell['comparison']=='dedicated_baseline' else plan['inputs']['baseline_profiles']
                    if key in batch_one_profiles and batch_one_profiles[key]!=b1:raise ValueError('Batch-one profiles differ across baselines')
                    batch_one_profiles[key]=b1
                from .dedicated_comparison import accounting
                cost=accounting(plan,p,observed['duration_s'])
                bindings[key]=bound;row.update(observed,resources=cost,
                    qualified_Q_per_declared_GPU=observed['observed_weighted_Q']/cost['declared_GPU_budget'] if plan['mode']=='gpu' else None)
            except (ValueError,KeyError,TypeError,OSError) as exc:row.update(status='invalid',reason=str(exc))
        rows.append(row)
    groups=[]
    for key in sorted(families):
        selected=[r for r in rows if r['family']==key];complete=all(r['status']=='passed' for r in selected)
        g=dict(family=key,complete=complete,status_counts=dict(__import__('collections').Counter(r['status'] for r in selected)),variants={},speedups=None,resources={},GPU_efficiency={})
        for v in sorted({r['variant'] for r in selected}):g['variants'][v]=stats([r['observed_weighted_Q'] for r in selected if r['variant']==v and r['status']=='passed'])
        for v in g['variants']:
            valid=[r for r in selected if r['variant']==v and r['status']=='passed']
            g['resources'][v]=[dict(repetition=r['cell']['repeat'],**r['resources']) for r in valid]
            g['GPU_efficiency'][v]=stats([r['qualified_Q_per_declared_GPU'] for r in valid]) if complete and valid and all(r['qualified_Q_per_declared_GPU'] is not None for r in valid) else None
        if complete:
            ratios={};rosa={r['cell']['repeat']:r['observed_weighted_Q'] for r in selected if r['variant']=='ROSA'}
            for v in sorted(set(g['variants'])-{'ROSA'}):
                pairs=[dict(repetition=r['cell']['repeat'],ratio=rosa[r['cell']['repeat']]/r['observed_weighted_Q'] if r['observed_weighted_Q']>0 else None) for r in selected if r['variant']==v]
                defined=all(r['ratio'] is not None for r in pairs)
                ratios[v]=dict(pairs=pairs,summary=stats([r['ratio'] for r in pairs]) if defined else None,complete=defined,
                    semantics='ROSA Q divided by baseline Q at explicitly reported budgets; CPU ratios are synthetic only')
            g['speedups']=ratios
        groups.append(g)
    return dict(schema_version='1.0',kind='r08_r09_paper_comparison',paper_sha256=paper['sha256'],rows=rows,groups=groups,
        status='complete' if all(g['complete'] for g in groups) else 'incomplete',paper_equivalence_claim=False,
        uncertainty_unit='independent declared repetitions, never individual frames/requests',
        gpu_results=all(b['mode']=='gpu' for b in bindings.values()) if bindings else False,
        limitations=['Fixed-budget baselines preserve overload outcomes; structural plan validation does not assert SLO feasibility.',
          'Same source, banks, protocol, model bindings and declared repetitions required. Missing/invalid cells suppress family speedups.',
          'Dedicated methods use explicit separate GPU budgets; reserved measurement GPU-seconds exclude setup and are not utilization or billed cost.',
          'CPU trials have zero actual GPU leases and no GPU-efficiency estimate; target slot counts are diagnostic only.',
          'Zero baseline qualified throughput leaves the ratio undefined; no infinite speedup is fabricated.'])
