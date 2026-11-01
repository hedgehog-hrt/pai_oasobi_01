"""Review-bound measured-profile admission. Hashes establish integrity, not trust.
All evidence stays under one portable root; no model loading or network access.
"""
from copy import deepcopy
import json
import csv,io
from pathlib import Path
from .execution_plan import digest,runtime_source_manifest,validate_execution_plan
from .regression import sha256
from .profile_runner import profile_candidate,verify_evidence
from .runtime_attestation import summarize_trace
from .r06_adoption import quality as r06_quality,review_checks as r06_review_checks,validate_review as validate_r06_review

CHECKS=('runtime_kernel_dtype_compile','workload_representative','queue_stability_and_slo',
        'gpu_background_load','heldout_quality','s1_task_success_and_batch_equivalence')
CRITICAL_PACKAGES={'torch','transformers','vllm','triton','flash-attn','numpy','pillow','ray','safetensors','tokenizers',
                   'flashinfer-python','nvidia-cutlass-dsl','nvidia-cudnn-frontend','cuda-bindings','cuda-python','pyarrow'}
IGNORED_ENV={'CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH','PYTHONDONTWRITEBYTECODE',
             'VLLM_TORCH_PROFILER_DIR','VLLM_TORCH_PROFILER_WITH_STACK','VLLM_TORCH_PROFILER_RECORD_SHAPES'}


def inside(root,name):
    root=Path(root).resolve();rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Evidence path must be root-relative')
    p=root/rel
    if any(x.is_symlink() for x in [p,*p.parents] if x!=root and x.is_relative_to(root)):
        raise ValueError('Evidence symlink is not allowed')
    if not p.resolve().is_relative_to(root):raise ValueError('Evidence path escapes root')
    return p


def read(p):return json.loads(Path(p).read_text())


def observed_signature(state):
    torch=deepcopy(state['observed_torch_state'])
    if not torch['cuda_available'] or not torch['devices']:raise ValueError('No observed CUDA runtime')
    for device in torch['devices']:
        device.pop('uuid',None);device.pop('logical_index',None)
    model=deepcopy(state.get('observed_model_state'))
    if model is not None:model.pop('parameter_devices',None)
    return dict(torch=torch,model=model)


def runtime_signature(environment,backend):
    """Ignore placement/profiler paths, retain critical versions and actual flags."""
    packages={k.lower().replace('_','-'):v for k,v in environment['packages'].items()}
    observed=observed_signature(backend.get('runtime',backend.get('attestation',{})))
    if not backend.get('model_artifacts_verified'):raise ValueError('Model artifact verification missing')
    return dict(python=environment['python'],python_binary_sha256=environment['python_binary_sha256'],
        system=environment['system'],machine=environment['machine'],
        packages={k:packages.get(k) for k in sorted(CRITICAL_PACKAGES)},
        selected_environment={k:v for k,v in environment['selected_environment'].items() if k not in IGNORED_ENV},
        gpu_measurement=backend.get('gpu_measurement'),**observed)


class Evidence:
    def __init__(self,root):self.root=Path(root).resolve();self.hashes={}
    def file(self,name):
        p=inside(self.root,name)
        if not p.is_file():raise ValueError('Missing evidence: '+str(name))
        self.hashes[str(p.relative_to(self.root))]=sha256(p);return p
    def json(self,name):return read(self.file(name))
    def tree(self,name):
        p=inside(self.root,name)
        if not p.is_dir():raise ValueError('Missing evidence directory: '+str(name))
        for f in sorted(p.rglob('*')):
            if f.is_symlink():raise ValueError('Evidence symlink is not allowed')
            if f.is_file():self.file(str(f.relative_to(self.root)))
        return p
    def sealed(self,name):
        p=self.tree(name);verify_evidence(p)
        if read(p/'source_manifest.json')!=runtime_source_manifest():raise ValueError('Evidence source differs from current runtime')
        return p


def _quality(e,request,c,corpus_id,signature):
    q=request['quality']
    if c['role']=='s1':
        # Current admission requires exact arrays; legacy summary-only evidence
        # remains historical and cannot certify this runtime revision.
        from .paired import load_state_bank
        from .action_regression import validate_run
        regression=e.tree(q['regression']);paired=e.tree(q['paired']);bank=e.tree(q['state_bank'])
        raw_regression=validate_run(regression)
        r=read(regression/'comparison_summary.json');server=read(regression/'server_evidence.json')
        if r['status']!='passed' or not r['gpu_execution_confirmed_by_response'] or not r['cases'] or r['cases']!=r['expected_cases']:
            raise ValueError('S1 numerical regression incomplete')
        cases=[read(p) for p in sorted(regression.glob('case-*.json'))]
        if cases!=r['results'] or len(cases)!=r['cases'] or not all(x['passed'] and x['same_loaded_model'] and x['default_rngs_restored'] for x in cases):
            raise ValueError('S1 numerical regression case mismatch')
        if server['checkpoint'].get('revision')!=c['model_revision'] or not server['checkpoint'].get('full_weight_hashes_verified'):
            raise ValueError('S1 regression checkpoint mismatch')
        observed=observed_signature(server['runtime'])
        if observed['model'] is None or observed!={k:signature[k] for k in ('torch','model')}:
            raise ValueError('S1 numerical regression runtime differs from profile')
        batch_regression=batch_diagnostic=None
        if c['batch_size']>1:
            batch_directory=e.tree(q['batch_regression']);batch_regression=validate_run(batch_directory,expected_batch_size=c['batch_size'])
            if read(batch_directory/'server_evidence.json')!=server:raise ValueError('S1 batch comparison uses a different loaded runtime')
            diagnostic_directory=e.tree(q['batch_diagnostic'])
            batch_diagnostic=validate_run(diagnostic_directory,expected_batch_size=c['batch_size'],require_passed=False,required_comparison='serial_vs_batch')
            if batch_diagnostic['status'] not in ('passed','mismatch') or batch_diagnostic['cases']!=batch_diagnostic['expected_cases']:
                raise ValueError('S1 cross-shape diagnostic incomplete')
            if read(diagnostic_directory/'server_evidence.json')!=server:raise ValueError('S1 diagnostic runtime differs')
            contexts=[read(path/'run_context.json') for path in (regression,batch_directory,diagnostic_directory)]
            controls=lambda x:dict(seed=x['seed'],atol=x['atol'],rtol=x['rtol'],inputs=[v['observation_sha256'] for v in x['inputs']])
            if any(controls(x)!=controls(contexts[0]) for x in contexts[1:]):raise ValueError('S1 regression/diagnostic inputs or controls differ')
        # Preserve all paired logs for review; verify completeness against the actual bank.
        b,_=load_state_bank(bank);p=read(paired/'paired_summary.json')
        if b['source'].get('kind')!='LIBERO_benchmark_initial_states' or read(paired/'input_manifest.json')!=b:
            raise ValueError('S1 state bank provenance mismatch')
        ids={x['case_id'] for x in b['cases']}
        if (p['execution_kind']!='libero_gpu' or not p['paired_comparison_valid'] or
            p['initial_states_manifest_sha256']!=sha256(bank/'initial_states.json') or
            p['started_episodes']!=p['intended_episodes'] or p['intended_episodes']!=2*len(ids) or
            len(p['pairs'])!=len(ids) or {x['case_id'] for x in p['pairs']}!=ids or
            not all(x['both_completed'] and x['initial_conditions_match'] for x in p['pairs'])):
            raise ValueError('S1 paired task evidence incomplete')
        outcomes=[read(x) for x in sorted(paired.glob('*/*/outcome.json'))]
        if len(outcomes)!=2*len(ids):raise ValueError('S1 paired outcomes missing')
        for case in b['cases']:
            pair=[x for x in outcomes if x['case_id']==case['case_id']]
            if len(pair)!=2 or {x['arm'] for x in pair}!={'official','shared'} or any(
                x['status']!='completed' or x['state_sha256']!=case['state_sha256'] or x['seed']!=case['seed'] for x in pair):
                raise ValueError('S1 paired case failed or mismatched')
            if any(pair[0].get(k) is None or pair[0][k]!=pair[1].get(k) for k in ('restored_state_sha256','initial_observation_sha256')):
                raise ValueError('S1 paired initial conditions differ')
        for arm in ('official','shared'):
            actual=dict(started=sum(x['arm']==arm for x in outcomes),completed=sum(x['arm']==arm and x['status']=='completed' for x in outcomes),
                successful=sum(x['arm']==arm and x['success'] for x in outcomes),errors=sum(x['arm']==arm and x['status']=='error' for x in outcomes))
            if p['arms'][arm]!=actual:raise ValueError('S1 task success summary differs from outcomes')
        # Bind the earlier harness to this exact backend through a human-authored association.
        association=e.json(q['association'])
        if association.get('config_sha256')!=digest(c) or association.get('corpus_id')!=corpus_id or association.get('source_manifest')!=runtime_source_manifest() or not all(
            isinstance(association.get(k),str) and association[k].strip() for k in ('reviewer','reason')):
            raise ValueError('S1 evidence association requires review')
        r06=r06_quality(e,q,c,server)
        return dict(kind='s1_regression_paired_and_r06_review',r06=r06,numerical_cases=len(cases),pairs=len(ids),
            arms=p['arms'],raw_action_tensors_recomputed=raw_regression['raw_arrays_recomputed'],batch_regression=batch_regression,batch_diagnostic=batch_diagnostic,
            numerical_scope='official and shared API at identical batch size/input/default seed; cross-shape comparison is descriptive',
            automatic_stochastic_batch_equivalence_claim=False,
            task_success_threshold='human review; no invented paper threshold')
    from .observation_quality import validate_quality_suite,evaluate_quality
    p=e.sealed(q['run']);s=read(p/'condition.json');bundle=read(p/'predictions.json')
    root=e.tree(q['corpus_root']);validate_quality_suite(s,root)
    if s['config']!=c or s['corpus']['corpus_id']!=corpus_id or s['split']!='holdout':raise ValueError('Quality config/corpus/holdout mismatch')
    rows=[json.loads(x) for x in (p/'requests.jsonl').read_text().splitlines()]
    if rows!=bundle['rows']:raise ValueError('Quality raw predictions differ')
    if {k:v for k,v in bundle.items() if k not in ('rows','bundle_id')}!=read(p/'result.json'):
        raise ValueError('Quality result differs from prediction bundle')
    if runtime_signature(bundle['worker_environment'],bundle['worker_after']['backend_attestation'])!=signature:
        raise ValueError('Quality runtime differs from profile')
    result=evaluate_quality(s,bundle,e.json(q['references']) if q.get('references') else None,
                            e.json(q['ratings']) if q.get('ratings') else None)
    if result['status']!='passed' or result['synthetic']:raise ValueError('Heldout quality has not passed')
    return result


def inspect(request,root):
    if set(request)!={'schema_version','performance','kernel','quality'} or request['schema_version']!='1.0':raise ValueError('Invalid adoption request')
    e=Evidence(root);paths=[e.sealed(x) for x in request['performance']]
    candidate=profile_candidate(paths)
    if candidate['status']=='blocked':raise ValueError('Performance blocked: '+','.join(candidate['blocking_reasons']))
    conditions=[read(p/'condition.json') for p in paths];c=conditions[0]['config'];corpus_id=conditions[0]['corpus_id']
    from .profile_protocol import validate_probe_config
    validate_probe_config(c)
    results=[read(p/'result.json') for p in paths]
    signature=runtime_signature(results[0]['worker_environment'],results[0]['worker_after']['backend_attestation'])
    if any(runtime_signature(r['worker_environment'],r['worker_after']['backend_attestation'])!=signature for r in results):
        raise ValueError('Profile runtimes differ')
    p=e.sealed(request['kernel']);k=read(p/'result.json');kc=read(p/'condition.json')
    if kc.get('config')!=c or kc.get('corpus_id')!=corpus_id or kc.get('split')!='evaluation' or kc.get('batch_size')!=c['batch_size']:
        raise ValueError('Kernel config/corpus/batch mismatch')
    if (k.get('status')!='complete' or k.get('kind')!='gpu_backend_kernel_diagnostic' or
        k.get('process_exit_code')!=0 or k.get('process_timed_out') is not False or k.get('process_interrupted') is not False or
        k.get('gpu_execution') is not True or k.get('pretrained_weights_loaded') is not True):raise ValueError('Kernel diagnostic did not complete on GPU')
    traces={name:summarize_trace(inside(p,name)) for name in k['traces']}
    if traces!=k['traces'] or not any(t['gpu_kernel_events_present'] for t in traces.values()):raise ValueError('Actual CUDA trace missing/mismatched')
    if runtime_signature(k['environment_after'],k['backend'])!=signature:raise ValueError('Kernel runtime differs from profile')
    nvidia=k['environment_after']['tools']['nvidia_smi']
    devices=k['backend'].get('runtime',k['backend'].get('attestation',{}))['observed_torch_state']['devices']
    physical={d['uuid'] for d in devices}
    telemetry=json.loads((paths[0]/'telemetry.jsonl').read_text().splitlines()[0])['sample']
    device_rows=[row for row in csv.reader(io.StringIO(nvidia['stdout']),skipinitialspace=True) if len(row)==4 and row[0].strip() in physical]
    if nvidia['status']!='ok' or len(device_rows)!=len(physical) or any(row[1].strip()!=c['hardware_kind'] or row[2].strip()!=telemetry['driver'] or float(row[3])!=telemetry['total_mib'] for row in device_rows):
        raise ValueError('Kernel GPU driver/hardware/memory evidence differs from profile')
    quality=_quality(e,request,c,corpus_id,signature)
    if c['role']!='s1':
        corpus=read(inside(root,request['quality']['run'])/'condition.json')['corpus']
        if kc.get('observation_sha256') not in {x['sha256'] for x in corpus['rows'] if x['split']=='evaluation'}:
            raise ValueError('Kernel input is not in the evaluation corpus')
    # profile_candidate includes absolute paths; replace only its provenance ID
    # with portable, root-relative evidence, preserving all measured numbers.
    performance=[dict(directory=name,evidence_sha256=sha256(inside(root,name)/'evidence.json')) for name in request['performance']]
    profile=deepcopy(candidate['profile']);profile['evidence_sha256']=digest(performance);profile['id']='reviewed-'+digest(performance)[:16]
    result=dict(schema_version='1.0',status='requires_review',request=deepcopy(request),profile=profile,
        config=c,corpus_id=corpus_id,source_manifest=runtime_source_manifest(),runtime_signature=signature,
        quality=quality,evidence_files=e.hashes,scope='reviewed profile admission; not a capacity, safety or paper reproduction certificate')
    result['evidence_set_sha256']=digest(result);return result


def review_template(inspection):
    result=dict(schema_version='1.0',evidence_set_sha256=inspection['evidence_set_sha256'],reviewer=None,
                checks={k:dict(accepted=None,reason=None) for k in CHECKS})
    if inspection['config']['role']=='s1':
        result['r06_checks']=r06_review_checks(inspection['quality']['r06'])
        result['s1_review_guidance']='Same-shape numerical agreement validates the adapter path only. Review all cross-shape differences and task evidence separately; unaligned seeds neither prove RNG causality nor stochastic/task equivalence.'
    return result


def adopt(request,root,review):
    result=inspect(request,root)
    if review.get('schema_version')!='1.0' or review.get('evidence_set_sha256')!=result['evidence_set_sha256']:
        raise ValueError('Review is stale or bound to different evidence')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip() or set(review.get('checks',{}))!=set(CHECKS):
        raise ValueError('Named reviewer and all checks required')
    for check in review['checks'].values():
        if check.get('accepted') is not True or not isinstance(check.get('reason'),str) or not check['reason'].strip():
            raise ValueError('Review incomplete or rejected')
    if result['config']['role']=='s1':validate_r06_review(result['quality']['r06'],review.get('r06_checks',{}))
    result.update(status='accepted',review=deepcopy(review));result['adoption_id']=digest(result);return result


def validate_adoption(value,root):
    expected=adopt(value['request'],root,value['review'])
    if value!=expected:raise ValueError('Adoption changed; regenerate after reviewing new evidence')
    return value


def validate_fleet_adoptions(plan,bundle):
    validate_execution_plan(plan)
    if plan['mode']=='cpu_dummy':return {}
    if not bundle or set(bundle)!={'evidence_root','adoptions'}:raise ValueError('GPU R07 requires reviewed profile adoptions')
    root=Path(bundle['evidence_root']);profiles={p['id']:p for p in plan['inputs']['fleet']['profiles']}
    required={w['profile_id'] for w in plan['workers']}
    if set(bundle['adoptions'])!=required:raise ValueError('Adoptions must cover exactly the deployed profiles')
    accepted={pid:validate_adoption(read(inside(root,name)),root) for pid,name in bundle['adoptions'].items()}
    for w in plan['workers']:
        a=accepted[w['profile_id']];c=a['config']
        if a['profile']!=profiles[w['profile_id']] or w['backend']!=c['binding'] or any(w[k]!=c[k] for k in
            ('component_id','role','model_revision','runtime_fingerprint','hardware_kind','batch_size','batch_wait_ms')):
            raise ValueError('Deployed profile/binding differs from accepted evidence')
        if len(w['members'])>c['streams']:raise ValueError('Deployment stream count exceeds profiled streams')
        for klass in plan['inputs']['fleet']['classes']:
            for model in klass['components']:
                if model['id']==c['component_id'] and model['deadline_ms']!=c['deadline_ms']:
                    raise ValueError('Deployment SLO differs from reviewed profile')
    return accepted


def import_profiles(fleet,adoptions,root):
    from .heterogeneous import Fleet
    result=deepcopy(fleet);result['profiles']=[validate_adoption(a,root)['profile'] for a in adoptions]
    Fleet.from_dict(result);return result
