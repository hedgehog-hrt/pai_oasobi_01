"""Bind dedicated methods to prospective paper cells, whole budgets and raw leases."""
from pathlib import Path
from .paper_conditions import validate as validate_paper
from .paper_baselines import _workload
from .execution_plan import validate_execution_plan,check_runtime_inventory
from .dedicated_baselines import prepare
from .dedicated_execution import compile_plan


def compile_cell(reference,paper,cell_id,profiles):
    validate_execution_plan(reference);validate_paper(paper)
    cells=[c for c in paper['cells'] if c['id']==cell_id]
    if len(cells)!=1 or cells[0].get('comparison')!='dedicated_baseline':raise ValueError('Expected a prospective dedicated paper cell')
    c=cells[0];_workload(reference,c)
    resources=prepare(reference,c['method'],profiles,gpu_budget=c['dedicated_gpu_budget'])
    if resources['required_physical_GPUs']!=c['required_physical_gpus']:raise ValueError('Whole workload dedicated count differs')
    # Infeasible cells are retained in the matrix; there is no runnable subset.
    return resources if resources['shortfall'] else compile_plan(resources)


def accounting(plan,root,duration_s):
    from .profile_adoption import read
    p=Path(root);before=read(p/'workers_before.json');after=read(p/'workers_after.json')
    checks=[check_runtime_inventory(plan,x) for x in (before,after)]
    if any((before[w]['instance_id'],before[w]['node_id'],before[w]['pid'],before[w]['gpu_ids'])!=
           (after[w]['instance_id'],after[w]['node_id'],after[w]['pid'],after[w]['gpu_ids']) for w in before):raise ValueError('Runtime/physical lease changed during fixed-allocation comparison')
    dedicated=plan.get('purpose')=='dedicated_GPU_execution'
    budget=plan['inputs']['resource_plan']['gpu_budget'] if dedicated else plan['inputs']['fleet']['hardware']['gpu_count']
    groups=len(plan['physical_groups']) if dedicated else len(plan['workers'])
    if groups>budget:raise ValueError('Live physical groups exceed declared budget')
    gpu=plan['mode']=='gpu'
    return dict(declared_GPU_budget=budget,target_physical_groups=groups,actual_GPU_leases=groups if gpu else 0,
        measurement_window_s=duration_s,declared_budget_GPU_seconds=budget*duration_s if gpu else 0,
        leased_GPU_seconds=groups*duration_s if gpu else 0,CPU_target_slot_seconds=groups*duration_s if not gpu else None,
        GPU_utilization=None,setup_GPU_seconds=None,
        semantics='reserved device-time during measurement, not measured compute utilization or total billed cost',
        inventory_checks=checks)


def prepare_family(reference,paper,family_key,profiles,banks,root,*,phase,controls=None,setup_budget_s=600.):
    """Materialize all plans/suites without starting Ray, models or a Pod."""
    from .paper_comparison import family
    from .paper_baselines import compile_baseline
    from .closed_loop_protocol import protocol,prepare_suite
    validate_paper(paper);validate_execution_plan(reference)
    cells=[c for c in paper['cells'] if c['experiment'] in ('R08','R09') and family(c)==family_key]
    if not cells:raise ValueError('Unknown paper family')
    controls=controls or {};rows=[]
    # Freeze timing across variants, including uncapped dedicated clients.
    base=protocol(reference,phase,**controls)
    fixed={k:base[k] for k in ('warmup_s','duration_s','request_timeout_s','sim_timeout_s','drain_timeout_s',
        'max_steps','execute_steps','max_episodes_per_robot','max_requests','max_capture_bytes')}
    fixed['warmup_min_success']=base['min_warmup_success_per_active_class_component']
    for c in cells:
        _workload(reference,c)
        plan=reference if c['comparison']=='ROSA' else compile_baseline(reference,paper,c['id'],profiles if c['comparison'] in ('full_baseline','dedicated_baseline') else None)
        if plan.get('kind')=='dedicated_GPU_resource_plan':rows.append(dict(cell=c,status='resource_infeasible',resources=plan,suite=None));continue
        pro=protocol(plan,phase,**fixed)
        s=prepare_suite(plan,banks,root,pro,repetitions=paper['inputs']['repetitions'],setup_budget_s=setup_budget_s)
        rows.append(dict(cell=c,status='prepared_unmeasured',plan=plan,suite=s))
    return dict(kind='prepared_paper_family',paper=paper,reference=reference,family=family_key,rows=rows,
        GPU_execution=False,models_started=False,paper_equivalence_claim=False)
