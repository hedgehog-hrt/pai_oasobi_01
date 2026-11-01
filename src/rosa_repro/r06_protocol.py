"""Fixed task/state cohorts; environment concurrency never multiplies the case set."""
from pathlib import Path
import random
from .paired import load_state_bank
from .execution_plan import digest,runtime_source_manifest
from .libero_bridge import LIBERO_COMMIT
from .regression import UPSTREAM_COMMIT

TASKS=dict(T01='pick_up_the_alphabet_soup_and_place_it_in_the_basket',
 T02='pick_up_the_tomato_sauce_and_place_it_in_the_basket',
 T03='pick_up_the_cream_cheese_and_place_it_in_the_basket')

def prepare(bank_root,*,mode,phase,selected_batch,server_evidence=None,seed=42,max_steps=720,
            execute_steps=8,call_timeout_s=120,trial_timeout_s=7200):
    if mode not in ('cpu_dummy','libero_gpu') or phase not in ('smoke','formal'):raise ValueError('Explicit mode/phase required')
    if type(selected_batch) is not int or selected_batch not in (1,2,4,8,16):raise ValueError('Invalid candidate batch')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid order seed')
    if any(type(v) is not int or v<1 for v in (max_steps,execute_steps,call_timeout_s,trial_timeout_s)) or execute_steps>max_steps:
        raise ValueError('Positive bounded budgets required')
    if execute_steps!=8 and mode=='libero_gpu':raise ValueError('Pinned action stride is eight simulation steps')
    tasks=list(TASKS) if phase=='formal' else ['T01'];banks={}
    for task in tasks:
        manifest,_=load_state_bank(Path(bank_root)/task)
        if manifest['task']!=TASKS[task]:raise ValueError('State bank task mismatch: '+task)
        if mode=='libero_gpu':
            source=manifest['source'];required=50 if phase=='formal' else 10
            if len(manifest['cases'])!=required:raise ValueError('Expected fixed 50 formal / 10 smoke initial states per task')
            if source.get('kind')!='LIBERO_benchmark_initial_states' or source.get('libero_commit')!=LIBERO_COMMIT or source.get('gr00t_commit')!=UPSTREAM_COMMIT:
                raise ValueError('Pinned LIBERO state provenance required')
        elif manifest['source'].get('kind')!='synthetic':raise ValueError('CPU fixture must be labelled synthetic')
        if len({(c['state_sha256'],c['seed']) for c in manifest['cases']})!=len(manifest['cases']):raise ValueError('Duplicate state/seed exposure')
        banks[task]=manifest
    if mode=='libero_gpu':
        if not server_evidence or server_evidence.get('model_revision_verified') is not True or not server_evidence.get('modality_config'):
            raise ValueError('Verified full-checkpoint server evidence and simulator modality config required')
    elif server_evidence is not None:raise ValueError('CPU fixture cannot impersonate model evidence')
    cells=[]
    for task in tasks:
        for width in ((1,2,4,selected_batch) if selected_batch>4 else (1,2,4)):
            for arm in ('official','shared_b1','selected'):
                cells.append(dict(id=f'{task}-{width}-{arm}',task=task,concurrency=width,arm=arm,
                    max_batch_size=selected_batch if arm=='selected' else 1))
    order=[c['id'] for c in cells];random.Random(seed).shuffle(order)
    suite=dict(source_manifest=runtime_source_manifest(),schema_version='1.0',kind='r06_fixed_state_regression',mode=mode,phase=phase,banks=banks,cells=cells,
        randomized_order=order,seed=seed,server_evidence=server_evidence,
        budgets=dict(max_steps=max_steps,execute_steps=execute_steps,call_timeout_s=call_timeout_s,trial_timeout_s=trial_timeout_s),
        batching='explicit_barrier_cohorts_no_padding',rng_alignment='unverified_across_batch_shapes',
        model_performance_claim=False,formal_LIBERO_claim_eligible=mode=='libero_gpu' and phase=='formal',
        candidate_batch_status='explicit_regression_candidate_not_automatic_profile_admission')
    suite['sha256']=digest(suite);return suite

def validate(suite,bank_root):
    candidate=next(c['max_batch_size'] for c in suite['cells'] if c['arm']=='selected')
    expected=prepare(bank_root,mode=suite['mode'],phase=suite['phase'],selected_batch=candidate,
        server_evidence=suite['server_evidence'],seed=suite['seed'],**suite['budgets'])
    if suite!=expected:raise ValueError('R06 design or bank evidence changed')
    return suite
