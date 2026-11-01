"""Paired T01 runner: explicit state bank, identical cases, all-started outcomes."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np

from .adapter import JsonlSink, MeasuredClient, check_observation_batch_one
from .regression import save_observation, sha256, write_json


def digest_value(value):
    """Content hash, independent of ZIP timestamps and dict insertion order."""
    def encode(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                return encode(x.tolist())
            return {'dtype':x.dtype.str,'shape':list(x.shape),
                    'bytes_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
        if isinstance(x, dict):
            return {k:encode(v) for k,v in sorted(x.items())}
        if isinstance(x, (list,tuple)):
            return [encode(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def save_state_bank(directory, states, *, task, source, seed=42):
    directory=Path(directory)
    if not states or type(seed) is not int or seed < 0:
        raise ValueError('Nonempty initial states and nonnegative seed required')
    arrays={}
    cases=[]
    for index,state in enumerate(states):
        a=np.asarray(state)
        if a.dtype.kind not in 'fiu' or a.ndim!=1 or a.size==0 or not np.isfinite(a).all():
            raise ValueError('Initial state must be a finite numeric vector')
        key=f'state_{index:04d}'
        arrays[key]=a
        cases.append(dict(case_id=f'T01-{index:04d}',initial_state_id=index,state_key=key,
                          state_sha256=digest_value(a),seed=seed+index))
    directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/'states.npz',**arrays)
    manifest=dict(schema_version='1.0',task=task,source=source,cases=cases,
                  states_file_sha256=sha256(directory/'states.npz'))
    write_json(directory/'initial_states.json',manifest)
    return manifest


def load_state_bank(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'initial_states.json').read_text())
    if manifest['schema_version']!='1.0' or sha256(directory/'states.npz')!=manifest['states_file_sha256']:
        raise ValueError('State bank version/hash mismatch')
    states={}
    seen=set()
    with np.load(directory/'states.npz',allow_pickle=False) as data:
        for case in manifest['cases']:
            cid=case['case_id']
            if not re.fullmatch(r'[A-Za-z0-9_-]+',cid) or cid in seen:
                raise ValueError('Invalid/duplicate case ID')
            seen.add(cid)
            a=data[case['state_key']]
            if (a.ndim!=1 or a.dtype.kind not in 'fiu' or not a.size or not np.isfinite(a).all()
                    or digest_value(a)!=case['state_sha256']):
                raise ValueError('State data/hash mismatch')
            if type(case['seed']) is not int or not 0<=case['seed']<2**32-100000:
                raise ValueError('Invalid case seed')
            states[cid]=a.copy()
    if not seen:
        raise ValueError('Empty state bank')
    return manifest,states


def run_paired(*, bank, output, env_factory, transport_factory, execution_kind,
               max_steps=720, execute_steps=8, capture_stride=1):
    """Factory contracts are shared by CPU tests and the pinned LIBERO bridge.

    env.reset(state, seed) -> (batched_observation, actual_restored_state_hash)
    env.step(action, requested_steps) -> dict(observation, steps, success, done)
    transport.reset(run_id,robot_id,episode_id); transport.predict(metadata,obs,seed)
    Request-wise seeds are a regression control, not a serving performance mode.
    """
    if any(type(v) is not int or v<1 for v in (max_steps,execute_steps,capture_stride)):
        raise ValueError('Positive step/capture counts required')
    if execution_kind not in ('cpu_dummy','libero_gpu'):
        raise ValueError('Explicit execution provenance required')
    manifest,states=load_state_bank(bank)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input_manifest.json',manifest)
    run_id='paired-'+uuid.uuid4().hex
    records=[]; outcomes=[]
    sink=JsonlSink(output/'events.jsonl')
    def emit(r): records.append(r); sink(r)
    started=time.monotonic_ns()
    try:
        for index,case in enumerate(manifest['cases']):
            # Alternate order without changing either arm's case/seed set.
            arms=('official','shared') if index%2==0 else ('shared','official')
            for arm in arms:
                case_dir=output/case['case_id']/arm
                case_dir.mkdir(parents=True,exist_ok=False)
                outcome=dict(case_id=case['case_id'],arm=arm,seed=case['seed'],
                    initial_state_id=case['initial_state_id'],state_sha256=case['state_sha256'],
                    started=True,success=False,status='error',termination_reason='exception',
                    executed_steps=0,action_chunks=0,initial_observation_sha256=None,
                    restored_state_sha256=None)
                env=transport=None
                try:
                    env=env_factory()
                    obs,restored=env.reset(states[case['case_id']].copy(),case['seed'])
                    check_observation_batch_one(obs)
                    outcome['restored_state_sha256']=restored
                    outcome['initial_observation_sha256']=digest_value(obs)
                    transport=transport_factory(arm)
                    client=MeasuredClient(run_id=run_id,robot_id=arm,reset_call=transport.reset,
                        predict_call=lambda metadata,observation:transport.predict(metadata,observation,
                                          case['seed']+metadata['observation_seq']),
                        sink=emit,clock_domain=run_id+'-client-monotonic')
                    client.reset(case['case_id'])
                    while outcome['executed_steps']<max_steps:
                        seq=outcome['action_chunks']
                        if seq%capture_stride==0:
                            save_observation(case_dir/f'observation-{seq:04d}',obs,dict(
                                case=case,arm=arm,task=manifest['task'],source=manifest['source'],
                                episode_id=case['case_id'],observation_seq=seq,
                                executed_steps=outcome['executed_steps'],execution_kind=execution_kind))
                        q=client.predict(obs)
                        if q is None:
                            raise RuntimeError('Inference rejected/cancelled; no action applied')
                        step_result={}
                        def execute(action):
                            limit=min(execute_steps,max_steps-outcome['executed_steps'])
                            np.savez_compressed(case_dir/f'action-{seq:04d}.npz',**action)
                            step_result.update(env.step(action,limit))
                            n=step_result['steps']
                            if type(n) is not int or not 1<=n<=limit:
                                raise ValueError('Environment reported invalid actual step count')
                            return n
                        n=client.apply(q,execute)
                        outcome['executed_steps']+=n; outcome['action_chunks']+=1
                        obs=step_result['observation']
                        if step_result['success'] or step_result['done']:
                            outcome.update(status='completed',success=bool(step_result['success']),
                                termination_reason='success' if step_result['success'] else 'environment_done')
                            break
                    else:
                        outcome.update(status='completed',termination_reason='step_limit')
                except Exception as exc:
                    outcome['error']=dict(type=type(exc).__name__,message=str(exc))
                finally:
                    for resource in (transport,env):
                        if resource is not None:
                            try: resource.close()
                            except Exception as exc:
                                outcome.setdefault('cleanup_errors',[]).append(type(exc).__name__+': '+str(exc))
                    outcomes.append(outcome)
                    write_json(case_dir/'outcome.json',outcome)
    finally:
        sink.close()
        ended=time.monotonic_ns()
        pairs=[]
        for case in manifest['cases']:
            by_arm={o['arm']:o for o in outcomes if o['case_id']==case['case_id']}
            valid=len(by_arm)==2 and all(o['status']=='completed' for o in by_arm.values())
            matched=valid and all(by_arm['official'][k] is not None and
                by_arm['official'][k]==by_arm['shared'][k] for k in
                ('state_sha256','restored_state_sha256','initial_observation_sha256','seed'))
            pairs.append(dict(case_id=case['case_id'],both_completed=valid,
                initial_conditions_match=matched,success_agreement=(
                by_arm['official']['success']==by_arm['shared']['success']) if valid else None))
        summary=dict(run_id=run_id,execution_kind=execution_kind,
            gpu_execution_requested=execution_kind=='libero_gpu',performance_benchmark=False,
            controlled_rng='per-case seed + request observation_seq, restored after each call',
            initial_states_manifest_sha256=sha256(Path(bank)/'initial_states.json'),
            started_ns=started,finished_ns=ended,clock_domain=run_id+'-client-monotonic',
            intended_episodes=2*len(manifest['cases']),started_episodes=len(outcomes),
            arms={arm:dict(started=sum(o['arm']==arm for o in outcomes),
                completed=sum(o['arm']==arm and o['status']=='completed' for o in outcomes),
                successful=sum(o['arm']==arm and o['success'] for o in outcomes),
                errors=sum(o['arm']==arm and o['status']=='error' for o in outcomes))
                for arm in ('official','shared')},pairs=pairs,
            paired_comparison_valid=len(outcomes)==2*len(pairs) and all(p['initial_conditions_match'] for p in pairs))
        write_json(output/'paired_summary.json',summary)
    return summary


def capture_initial_observations(bank,output,env_factory):
    """Capture reset observations without issuing policy/action requests."""
    manifest,states=load_state_bank(bank)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for case in manifest['cases']:
        env=env_factory()
        try:
            obs,restored=env.reset(states[case['case_id']].copy(),case['seed'])
            save_observation(output/case['case_id'],obs,dict(case=case,task=manifest['task'],
                source=manifest['source'],restored_state_sha256=restored,observation_seq=0,executed_steps=0))
        finally:env.close()
    write_json(output/'capture_summary.json',dict(cases=len(manifest['cases']),policy_calls=0,executed_steps=0,
        initial_states_manifest_sha256=sha256(Path(bank)/'initial_states.json')))
