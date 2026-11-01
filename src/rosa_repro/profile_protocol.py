"""Post-Pod probe definitions, disjoint saved observations and read-only preflight.
No imports of Ray or model loaders are required for preparing these artifacts.
"""
from copy import deepcopy
from importlib import metadata
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys

from .contracts import obj,TEXT,POSITIVE,SHA256,validate,load_json
from .execution_plan import digest,runtime_source_manifest,validate_binding,DUMMY_FACTORY
from .regression import load_observation,sha256,MODEL_REVISION
from .paired import digest_value
from .vlm_contract import default_config,runtime_fingerprint,validate_config
from .profiling_budget import formal_timing,default_timing_policy,TIMING_SCHEMA

CONFIG_SCHEMA=obj(dict(schema_version={'const':'1.0'},mode={'enum':['gpu','cpu_dummy']},
    component_id=TEXT,role={'enum':['s1','s2','safety','monitor']},model_revision=TEXT,runtime_fingerprint=TEXT,
    hardware_kind=TEXT,binding={'type':'object'},batch_size={'type':'integer','minimum':1,'maximum':64},
    batch_wait_ms={'type':'number','minimum':0,'maximum':1000},streams={'type':'integer','minimum':1,'maximum':256},
    deadline_ms=POSITIVE,request_timeout_s={'type':'number','exclusiveMinimum':0,'maximum':600},
    drain_timeout_s={'type':'number','exclusiveMinimum':0,'maximum':600},
    max_dispatch_lag_s={'type':'number','minimum':0,'maximum':10},
    payload_overlay=obj(dict(plan=TEXT,context=obj(dict(execution_history=TEXT,previous_plan=TEXT,task_feedback=TEXT),[])),[])))


def template(role,engine='vllm'):
    if role not in ('s1','s2','safety','monitor'):raise ValueError('Unknown role')
    if role=='s1':
        if engine!='groot':raise ValueError('S1 template requires groot')
        binding=dict(factory='rosa_repro.model_backends:groot_backend',kwargs=dict(repo='/SET/GR00T_REPO',
            model_path='/SET/GR00T_CHECKPOINT',instruction_key='task',stateless_verified=False))
        revision=MODEL_REVISION;fp='SET_VERIFIED_GROOT_RUNTIME_FINGERPRINT'
    else:
        cfg=default_config(role,engine)
        binding=dict(factory='rosa_repro.vlm_backends:'+('vllm_vlm_backend' if engine=='vllm' else 'transformers_vlm_backend'),
            kwargs=dict(config=cfg,model_path='/SET/QWEN_'+('3B' if role=='safety' else '7B')+'_SNAPSHOT'))
        revision=cfg['revision'];fp=runtime_fingerprint(cfg)
    binding['kwargs']['cuda_measurement']=True
    return dict(schema_version='1.0',mode='gpu',component_id=role,role=role,model_revision=revision,
        runtime_fingerprint=fp,hardware_kind='SET_ACTUAL_GPU_NAME',binding=binding,batch_size=1,batch_wait_ms=0,
        streams=16,deadline_ms={'s1':200,'s2':2000,'safety':500,'monitor':2000}[role],
        request_timeout_s=45.,drain_timeout_s=60.,max_dispatch_lag_s=.05,payload_overlay={})


def validate_probe_config(c,*,ready=True):
    validate(c,CONFIG_SCHEMA);validate_binding(c['binding'],c['mode'])
    if c['streams']<c['batch_size']:raise ValueError('Streams must cover requested batch size')
    if ready and 'SET_' in json.dumps(c):raise ValueError('Template still contains unresolved SET_ fields')
    if ready and '/SET/' in json.dumps(c):raise ValueError('Template still contains unresolved paths')
    if c['mode']=='gpu':
        b=c['binding'];kw=b['kwargs']
        if c['role']=='s1':
            if b['factory']!='rosa_repro.model_backends:groot_backend' or c['model_revision']!=MODEL_REVISION:
                raise ValueError('S1 requires the pinned GR00T factory/revision')
            if ready and kw.get('stateless_verified') is not True:raise ValueError('GR00T stateless regression is unresolved')
        else:
            cfg=validate_config(kw['config']);expected='rosa_repro.vlm_backends:'+('vllm_vlm_backend' if cfg['engine']=='vllm' else 'transformers_vlm_backend')
            if (b['factory'],c['role'],c['model_revision'],c['runtime_fingerprint'])!=(expected,cfg['role'],cfg['revision'],runtime_fingerprint(cfg,kw.get('runtime_variant'))):
                raise ValueError('VLM config identity mismatch')
            if c['batch_size']!=1 or c['batch_wait_ms']!=0:raise ValueError('VLM admission requires batch 1 and zero wait')
            if c['request_timeout_s']<=cfg['generation_timeout_s']:raise ValueError('Client timeout must exceed generation timeout')
        for key in ('model_path','repo'):
            if key in kw and not Path(kw[key]).is_absolute():raise ValueError('Use absolute model/repository paths')
    return c


def probe_plan(config):
    c=validate_probe_config(config)
    w=dict(worker_id='probe-worker',component_id=c['component_id'],role=c['role'],model_revision=c['model_revision'],
        runtime_fingerprint=c['runtime_fingerprint'],hardware_kind=c['hardware_kind'],expected_memory_mib=1,
        profile_id='unprofiled-probe',profile_source='unmeasured',batch_size=c['batch_size'],batch_wait_ms=c['batch_wait_ms'],
        resources=dict(num_cpus=1,num_gpus=int(c['mode']=='gpu')),members=[f'probe-{i}' for i in range(c['streams'])],backend=deepcopy(c['binding']))
    result=dict(purpose='single_component_calibration_not_R07_schedule',config=deepcopy(c),mode=c['mode'],workers=[w])
    result['plan_id']=digest(result)
    return result


def validate_probe_plan(p):
    if p!=probe_plan(p['config']):raise ValueError('Probe plan modified')
    return p


def build_corpus(root,*,kind='saved_robot_observations',seed=42):
    if kind not in ('saved_robot_observations','synthetic_cpu'):raise ValueError('Unknown corpus provenance')
    root=Path(root).resolve();rows=[];groups=set();declared={}
    for path in sorted(root.rglob('observation.json')):
        obs,m=load_observation(path.parent);p=m['provenance']
        # Both arms/frames of the same initial state stay in the same split.
        key=digest(dict(task=p['task'],state_sha256=p['case']['state_sha256']))
        validate(p['case']['state_sha256'],SHA256);groups.add(key)
        rows.append(dict(directory=str(path.parent.relative_to(root)),sha256=digest_value(obs),
            manifest_sha256=sha256(path),group_id=key))
        if 'capture_split' in p:
            if p['capture_split'] not in ('profile','evaluation','holdout'):raise ValueError('Invalid declared capture split')
            declared[rows[-1]['directory']]=p['capture_split']
    if len(groups)<2:raise ValueError('Need at least two distinct task/initial-state groups')
    order=sorted(groups);random.Random(seed).shuffle(order);profile=set(order[:len(order)//2])
    if declared and len(declared)!=len(rows):raise ValueError('Cannot mix declared and automatically assigned splits')
    assigned={}
    for row in rows:
        row['split']=declared[row['directory']] if declared else 'profile' if row['group_id'] in profile else 'evaluation'
        if row['group_id'] in assigned and assigned[row['group_id']]!=row['split']:raise ValueError('State group split leakage')
        assigned[row['group_id']]=row['split']
    c=dict(schema_version='1.0',kind=kind,seed=seed,rows=rows)
    c['corpus_id']=digest(c)
    return c


def load_corpus(manifest,root,split,mode):
    c=load_json(manifest) if isinstance(manifest,(str,Path)) else deepcopy(manifest)
    expected=dict(c);expected.pop('corpus_id',None)
    if c.get('schema_version')!='1.0' or c.get('corpus_id')!=digest(expected):raise ValueError('Corpus hash/version mismatch')
    if c.get('kind') not in ('synthetic_cpu','saved_robot_observations'):raise ValueError('Unknown corpus kind')
    if mode=='gpu' and c['kind']!='saved_robot_observations':raise ValueError('GPU run refuses synthetic corpus')
    if split not in ('profile','evaluation','holdout'):raise ValueError('Unknown split')
    groups={};seen=set();samples=[];root=Path(root).resolve()
    for row in c['rows']:
        if row['directory'] in seen:raise ValueError('Duplicate corpus path')
        seen.add(row['directory'])
        if row['split'] not in ('profile','evaluation','holdout'):raise ValueError('Invalid split')
        if row['group_id'] in groups and groups[row['group_id']]!=row['split']:raise ValueError('Initial-state leakage between splits')
        groups[row['group_id']]=row['split']
        directory=(root/row['directory']).resolve()
        if not directory.is_relative_to(root):raise ValueError('Corpus path escapes root')
        obs,m=load_observation(directory)
        if sha256(directory/'observation.json')!=row['manifest_sha256'] or digest_value(obs)!=row['sha256']:
            raise ValueError('Saved observation hash mismatch')
        p=m['provenance']
        if row['group_id']!=digest(dict(task=p['task'],state_sha256=p['case']['state_sha256'])):raise ValueError('Observation group mismatch')
        if mode=='gpu' and p.get('execution_kind')=='cpu_dummy':raise ValueError('CPU observation cannot be relabelled as real')
        if p.get('capture_split',row['split'])!=row['split']:raise ValueError('Capture split was reassigned')
        if row['split']==split:samples.append(dict(observation=obs,sha256=row['sha256'],group_id=row['group_id'],
            sample_id=digest(row),directory=row['directory']))
    if not samples:raise ValueError('Requested corpus split is empty')
    return c,samples


def phase_protocol(phase):
    if phase=='formal':return dict(warmup_min_s=30.,warmup_max_s=180.,warmup_min_success=20,
        measure_min_s=120.,measure_max_s=900.,measure_min_success=1000,max_requests=100000)
    if phase=='exploratory':return dict(warmup_min_s=5.,warmup_max_s=30.,warmup_min_success=5,
        measure_min_s=10.,measure_max_s=60.,measure_min_success=10,max_requests=10000)
    if phase=='cpu_smoke':return dict(warmup_min_s=.03,warmup_max_s=.5,warmup_min_success=2,
        measure_min_s=.12,measure_max_s=.8,measure_min_success=4,max_requests=200)
    raise ValueError('Unknown phase')


def make_suite(config,corpus,*,stage,phase,rates=(),repetitions=1,timing_policy=None,budget_limit_s=None):
    validate_probe_config(config)
    if stage not in ('smoke','pilot','profile'):raise ValueError('Unknown stage')
    if phase=='cpu_smoke' and config['mode']!='cpu_dummy':raise ValueError('CPU smoke cannot run GPU')
    if phase=='formal' and stage!='profile':raise ValueError('Formal protocol only applies to profile stage')
    if timing_policy is not None and phase!='formal':raise ValueError('Timing policy is only for formal profiles')
    if budget_limit_s is not None and (type(budget_limit_s) not in (int,float) or not math.isfinite(budget_limit_s) or budget_limit_s<=0):
        raise ValueError('Positive finite suite budget required')
    policy=deepcopy(timing_policy or default_timing_policy()) if phase=='formal' else None
    if policy is not None:validate(policy,TIMING_SCHEMA)
    if type(repetitions) is not int or not 1<=repetitions<=10:raise ValueError('Invalid repetition count')
    if stage=='profile':
        if not rates or len(set(rates))!=len(rates) or len(rates)>50 or any(type(r) not in (int,float) or not math.isfinite(r) or not 0<r<=10000 for r in rates):
            raise ValueError('Explicit, distinct positive profile rates required')
    elif rates:raise ValueError('Smoke/pilot uses bounded closed-loop requests, not offered rates')
    if corpus['kind']=='synthetic_cpu' and config['mode']=='gpu':raise ValueError('Synthetic corpus refused')
    cases=[]
    for rate in (rates if stage=='profile' else [None]):
        for rep in range(repetitions):
            c=dict(stage=stage,phase=phase,offered_hz=rate,rep=rep,protocol=phase_protocol(phase))
            if phase=='formal':
                c['timing']=formal_timing(rate,config,policy);c['protocol']=c['timing']['protocol']
            c['case_id']=stage+'-'+digest(c)[:16];cases.append(c)
    total=sum(c['timing']['bounded_phase_and_drain_s'] if 'timing' in c else
              c['protocol']['warmup_max_s']+c['protocol']['measure_max_s']+2*(config['drain_timeout_s']+2*config['request_timeout_s'])+5 for c in cases)
    blocked=[dict(case_id=c['case_id'],reasons=c['timing']['blocking_reasons']) for c in cases if 'timing' in c and not c['timing']['executable']]
    if budget_limit_s is not None and total>budget_limit_s:blocked.append(dict(case_id=None,reasons=['suite_time_budget_exceeded']))
    return dict(schema_version='1.0',config=deepcopy(config),corpus=deepcopy(corpus),
        split='evaluation' if stage=='smoke' else 'profile',cases=cases,source_manifest=runtime_source_manifest(),
        timing_policy=policy,budget_limit_s=budget_limit_s,executable=not blocked,unresolved=blocked,estimated_max_trial_s=total,
        budget_excludes='model load/file hashing, Ray scheduling, resets/environment probes and process teardown; no monetary estimate')


def validate_suite(s):
    cases=s['cases']
    if not cases:raise ValueError('Empty suite')
    first=cases[0];rates=list(dict.fromkeys(c['offered_hz'] for c in cases)) if first['stage']=='profile' else []
    reps=max(c['rep'] for c in cases)+1
    expected=make_suite(s['config'],s['corpus'],stage=first['stage'],phase=first['phase'],rates=rates,repetitions=reps,
                        timing_policy=s.get('timing_policy'),budget_limit_s=s.get('budget_limit_s'))
    if s!=expected:raise ValueError('Suite/source/config changed; prepare a new suite')
    return s


def environment_evidence():
    from .runtime_attestation import static_environment
    return static_environment()


def gpu_telemetry(properties):
    if properties is None:return dict(kind='cpu_no_gpu',gpu_execution=False)
    uuid=properties.get('uuid','')
    # PyTorch 2.9 exposes the physical UUID without NVIDIA's "GPU-" prefix.
    # Preserve the worker's identity in the result, but query/verify the complete
    # NVIDIA identifier. Never fall back to a potentially remapped device index.
    import re
    if not isinstance(uuid,str):return dict(kind='unavailable',error='Physical GPU UUID unavailable')
    query_uuid='GPU-'+uuid if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid) else uuid
    if not query_uuid.startswith('GPU-'):return dict(kind='unavailable',error='Physical GPU UUID unavailable')
    try:
        raw=subprocess.run(['nvidia-smi','--id='+query_uuid,
            '--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True).stdout
        import csv
        rows=list(csv.reader(raw.strip().splitlines()))
        if len(rows)!=1 or len(rows[0])!=6 or rows[0][0].strip()!=query_uuid:raise ValueError('GPU telemetry identity mismatch')
        row=[s.strip() for s in rows[0]]
        numbers=[float(row[i]) for i in (3,4,5)]
        if not all(math.isfinite(x) and x>=0 for x in numbers):raise ValueError('Invalid telemetry values')
        return dict(kind='nvidia_smi_sample',uuid=uuid,nvidia_smi_uuid=row[0],name=row[1],driver=row[2],total_mib=numbers[0],
            used_mib=numbers[1],utilization_pct=numbers[2],scope='whole device sample, including other processes; not allocation peak')
    except Exception as exc:return dict(kind='unavailable',error_type=type(exc).__name__)


def preflight(config):
    c=validate_probe_config(config);report=dict(config_sha256=digest(c),environment=environment_evidence(),gpu_model_loaded=False)
    if c['mode']=='cpu_dummy':return dict(report,status='passed',scope='CPU harness only')
    expected=c['binding'].get('runtime_env',{}).get('py_executable')
    if expected and Path(expected).resolve()!=Path(sys.executable).resolve():raise ValueError('Run preflight with the configured worker Python')
    kw=c['binding']['kwargs']
    if c['role']=='s1':
        from .regression import verify_upstream
        from .checkpoint_verification import verify_groot_checkpoint
        report.update(upstream=verify_upstream(kw['repo']),checkpoint=verify_groot_checkpoint(kw['model_path']))
    else:
        from .vlm_backends import check_versions
        from .vlm_contract import verify_snapshot
        variant=kw.get('runtime_variant')
        report.update(versions=check_versions(kw['config']['engine'],variant),runtime_variant=variant,
            checkpoint=verify_snapshot(kw['model_path'],kw['config']['model_id']))
    return dict(report,status='passed',scope='local dependencies and file hashes; no model/GPU load')
