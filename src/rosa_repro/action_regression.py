"""Persist exact S1 input/action arrays and recompute numerical comparisons.
A seed shared across different batch shapes does not align per-sample random noise.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .regression import seeded_rng,compare_actions,save_observation,load_observation,write_json,sha256
from .batching import predict_policy_batch
from .adapter import check_observation_batch_one,check_action_batch_one
from .paired import digest_value
from .execution_plan import digest,runtime_source_manifest
from .profile_adoption import inside,read


def comparison_controls(n,comparison):
    if comparison=='same_shape':return 'official_same_batch_vs_shared','identical_seed_and_shape'
    if comparison=='serial_vs_batch' and n>1:return 'serial_reference_vs_shared_batch','unverified_across_batch_shapes'
    raise ValueError('Use same_shape, or serial_vs_batch with batch size > 1')


def official_batch_reference(policy,observations):
    """Independent packing/unpacking of the pinned public batched API.
    Never call the shared combine/split helper on the reference side.
    """
    n=len(observations)
    if n==1:
        result,_=policy.get_action(deepcopy(observations[0]));check_action_batch_one(result);return [deepcopy(result)]
    keys=set(observations[0]);batch={}
    if any(set(o)!=keys for o in observations):raise ValueError('Reference observation keys differ')
    for key in sorted(keys):
        values=[o[key] for o in observations]
        if key.startswith(('video.','state.')):
            if any(v.shape!=values[0].shape or v.dtype!=values[0].dtype for v in values):raise ValueError('Reference shape/dtype differs')
            batch[key]=np.stack([v[0] for v in values],axis=0)
        else:batch[key]=[v if isinstance(v,str) else v[0] for v in values]
    action,_=policy.get_action(batch)
    if not isinstance(action,dict) or not action or any(not isinstance(v,np.ndarray) or v.ndim!=3 or v.shape[0]!=n for v in action.values()):
        raise ValueError('Invalid official batched action')
    rows=[{key:np.take(v,[i],axis=0) for key,v in action.items()} for i in range(n)]
    for row in rows:check_action_batch_one(row)
    return rows


def collect_arrays(policy,observations,*,seed,atol=0.,rtol=0.,comparison='same_shape'):
    n=len(observations)
    if not 1<=n<=64 or type(seed) is not int or not 0<=seed<2**32-n:raise ValueError('Invalid batch/seed controls')
    for obs in observations:check_observation_batch_one(obs)
    # Validate tolerances before any model call.
    if not np.isfinite([atol,rtol]).all() or min(atol,rtol)<0:raise ValueError('Invalid tolerance')
    mode,alignment=comparison_controls(n,comparison)
    if comparison=='same_shape':
        with seeded_rng(seed):references=official_batch_reference(policy,observations)
    else:
        references=[]
        for i,obs in enumerate(observations):
            with seeded_rng(seed+i):action,_=policy.get_action(deepcopy(obs))
            check_action_batch_one(action);references.append(deepcopy(action))
    with seeded_rng(seed):candidates=predict_policy_batch(policy,deepcopy(observations))
    comparisons=[compare_actions(a,b,atol=atol,rtol=rtol) for a,b in zip(references,candidates)]
    return dict(schema_version='2.0',comparison=comparison,mode=mode,
        batch_size=n,seed=seed,atol=atol,rtol=rtol,same_loaded_model=True,default_rngs_restored=True,
        rng_alignment=alignment,
        input_sha256=[digest_value(o) for o in observations],passed=all(x['passed'] for x in comparisons),comparisons=comparisons,
        reference_actions=references,candidate_actions=candidates)


def _summary(response):return {k:deepcopy(v) for k,v in response.items() if k not in ('reference_actions','candidate_actions')}


def save_case(directory,observations,response,*,provenance,server_evidence):
    out=Path(directory)
    n=len(observations);refs=response['reference_actions'];candidates=response['candidate_actions']
    if response['batch_size']!=n or len(refs)!=n or len(candidates)!=n or len(provenance)!=n:raise ValueError('Regression batch attribution mismatch')
    if response['input_sha256']!=[digest_value(o) for o in observations]:raise ValueError('Regression response input differs')
    comparisons=[compare_actions(a,b,atol=response['atol'],rtol=response['rtol']) for a,b in zip(refs,candidates)]
    if comparisons!=response['comparisons'] or response['passed']!=all(x['passed'] for x in comparisons):raise ValueError('Server comparison differs from raw arrays')
    expected_mode,expected_rng=comparison_controls(n,response['comparison'])
    if response.get('schema_version')!='2.0':raise ValueError('Updated comparison response required')
    if response['mode']!=expected_mode or response['rng_alignment']!=expected_rng or response['same_loaded_model'] is not True or response['default_rngs_restored'] is not True:
        raise ValueError('Regression controls differ')
    out.mkdir(parents=True,exist_ok=False)
    for i,(obs,ref,candidate) in enumerate(zip(observations,refs,candidates)):
        save_observation(out/f'input-{i:03d}',obs,provenance[i])
        np.savez_compressed(out/f'reference-{i:03d}.npz',**ref);np.savez_compressed(out/f'candidate-{i:03d}.npz',**candidate)
    write_json(out/'server_evidence.json',server_evidence);write_json(out/'comparison.json',_summary(response))
    hashes={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest=dict(schema_version='2.0',kind='s1_exact_array_regression',batch_size=n,files=hashes,source_manifest=runtime_source_manifest())
    manifest['case_id']=digest(manifest);write_json(out/'case.json',manifest);return manifest


def reaudit_case(directory):
    out=Path(directory);m=read(out/'case.json');base=dict(m);base.pop('case_id',None)
    if m['schema_version'] not in ('1.0','2.0') or m['kind']!='s1_exact_array_regression' or m['case_id']!=digest(base):raise ValueError('Raw regression manifest changed')
    n=m['batch_size']
    if type(n) is not int or not 1<=n<=64:raise ValueError('Invalid raw batch size')
    required={'server_evidence.json','comparison.json'}
    for i in range(n):required.update({f'input-{i:03d}/observation.json',f'input-{i:03d}/arrays.npz',f'reference-{i:03d}.npz',f'candidate-{i:03d}.npz'})
    if set(m['files'])!=required:raise ValueError('Raw regression evidence coverage differs')
    for name,h in m['files'].items():
        p=inside(out,name)
        if not p.is_file() or sha256(p)!=h:raise ValueError('Raw regression missing/modified: '+name)
    r=read(out/'comparison.json');observations=[];comparisons=[]
    mode,alignment=comparison_controls(n,r['comparison']) if m['schema_version']=='2.0' else (
        'official_vs_shared_batch_one' if n==1 else 'serial_reference_vs_shared_batch','identical_seed_and_shape' if n==1 else 'unverified_across_batch_shapes')
    if (r['mode']!=mode or r['rng_alignment']!=alignment or r['schema_version']!=m['schema_version'] or
        r['same_loaded_model'] is not True or r['default_rngs_restored'] is not True):raise ValueError('Saved regression controls differ')
    for i in range(n):
        obs,_=load_observation(out/f'input-{i:03d}');observations.append(obs)
        with np.load(out/f'reference-{i:03d}.npz',allow_pickle=False) as a,np.load(out/f'candidate-{i:03d}.npz',allow_pickle=False) as b:
            comparisons.append(compare_actions(dict(a),dict(b),atol=r['atol'],rtol=r['rtol']))
    if r['batch_size']!=n or r['comparisons']!=comparisons or r['input_sha256']!=[digest_value(o) for o in observations] or r['passed']!=all(x['passed'] for x in comparisons):
        raise ValueError('Saved numerical summary differs from arrays/inputs')
    result=dict(case_id=m['case_id'],batch_size=n,passed=all(x['passed'] for x in comparisons),comparisons=comparisons,
        rng_alignment=r['rng_alignment'],same_loaded_model=r['same_loaded_model'],default_rngs_restored=r['default_rngs_restored'],
        seed=r['seed'],atol=r['atol'],rtol=r['rtol'],source_manifest=m['source_manifest'],server_evidence=read(out/'server_evidence.json'))
    if m['schema_version']=='2.0':result['comparison']=r['comparison']
    return result


def seal_run(directory):
    root=Path(directory);files={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='regression_run.json'}
    write_json(root/'regression_run.json',dict(schema_version='1.0',files=files,sha256=digest(files)))


def validate_run(directory,*,current_source=True,expected_batch_size=1,require_passed=True,required_comparison=None):
    out=Path(directory);s=read(out/'comparison_summary.json');server=read(out/'server_evidence.json');context=read(out/'run_context.json')
    index=read(out/'regression_run.json')
    if index['sha256']!=digest(index['files']):raise ValueError('Regression run manifest changed')
    if set(index['files'])!={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='regression_run.json'}:raise ValueError('Regression run file coverage changed')
    for name,h in index['files'].items():
        if sha256(inside(out,name))!=h:raise ValueError('Regression run missing/modified file: '+name)
    if s.get('schema_version') not in ('2.0','3.0') or s['status'] not in ('passed','mismatch','failed') or not 0<s['cases']<=s['expected_cases']:
        raise ValueError('Complete raw-array regression required')
    kind=context.get('comparison','same_shape' if expected_batch_size==1 else 'serial_vs_batch')
    _,alignment=comparison_controls(expected_batch_size,kind)
    if (require_passed and kind!='same_shape') or (required_comparison is not None and kind!=required_comparison):
        raise ValueError('Same-shape regression required for numerical admission; cross-shape comparison is diagnostic only')
    if s['schema_version']=='3.0' and (s.get('comparison')!=kind or s.get('numerical_admission_eligible')!=(kind=='same_shape' and s['status']=='passed')):
        raise ValueError('Comparison purpose/admission flags differ')
    if require_passed and (s['status']!='passed' or s['cases']!=s['expected_cases']):raise ValueError('Complete raw-array regression required')
    files=sorted(out.glob('case-*.json'));rows=[read(x) for x in files]
    if len(rows)!=s['cases'] or rows!=s['results'] or context['expected_cases']!=s['expected_cases'] or context['batch_size']!=expected_batch_size or len(context['inputs'])!=s['expected_cases']*expected_batch_size:
        raise ValueError('Regression case coverage differs')
    computed='passed' if all(x['passed'] for x in rows) else 'mismatch'
    if s['status']=='failed':
        if not s['error']:raise ValueError('Failed capture requires error evidence')
    elif s['cases']!=s['expected_cases'] or s['status']!=computed or s['error'] is not None:raise ValueError('Regression status differs from cases')
    if current_source and server.get('rosa_source_manifest')!=runtime_source_manifest():raise ValueError('Regression server source differs')
    if current_source and context.get('source_manifest')!=runtime_source_manifest():raise ValueError('Regression run source differs')
    seen=set()
    for index,row in enumerate(rows):
        raw=inside(out,row['raw_case_directory']);m=read(raw/'case.json')
        if row['raw_case_directory'] in seen or row['raw_case_sha256']!=sha256(raw/'case.json'):raise ValueError('Duplicate or changed raw case')
        seen.add(row['raw_case_directory']);r=reaudit_case(raw)
        if r.get('comparison','same_shape' if expected_batch_size==1 else 'serial_vs_batch')!=kind or r['rng_alignment']!=alignment:raise ValueError('Mixed comparison kinds')
        saved_inputs=read(raw/'comparison.json')['input_sha256']
        if saved_inputs!=[x['observation_sha256'] for x in context['inputs'][index*expected_batch_size:(index+1)*expected_batch_size]]:
            raise ValueError('Regression raw inputs differ from planned inputs')
        if (r['seed'],r['atol'],r['rtol'])!=(context['seed']+index*expected_batch_size,context['atol'],context['rtol']):raise ValueError('Run numerical controls differ')
        if r['server_evidence']!=server or r['batch_size']!=expected_batch_size or (require_passed and r['passed'] is not True):raise ValueError('Regression batch/runtime/match differs')
        if current_source and r['source_manifest']!=runtime_source_manifest():raise ValueError('Raw regression client source differs')
        if row['comparison']!={k:v for k,v in r.items() if k not in ('source_manifest','server_evidence')}:raise ValueError('Run summary differs from raw comparison')
        if any(row[k]!=r[k] for k in ('passed','same_loaded_model','default_rngs_restored')):raise ValueError('Case flags differ from raw comparison')
    return dict(status=s['status'],cases=len(rows),expected_cases=s['expected_cases'],batch_size=expected_batch_size,raw_arrays_recomputed=True,
        comparison=kind,rng_alignment=alignment,numerical_admission_eligible=kind=='same_shape' and s['status']=='passed',
        automatic_stochastic_batch_equivalence_claim=False)
