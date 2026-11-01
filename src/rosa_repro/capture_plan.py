"""Bounded LIBERO observation collection with preassigned state-disjoint splits."""
from copy import deepcopy
import math,random,uuid
from pathlib import Path
import numpy as np
from .contracts import obj,validate
from .execution_plan import digest,runtime_source_manifest
from .paired import load_state_bank
from .regression import sha256,write_json,save_observation
from .adapter import JsonlSink,MeasuredClient

DESIGN_SCHEMA=obj(dict(schema_version={'const':'1.0'},seed={'type':'integer','minimum':0,'maximum':2**31-1},
    states_per_task={'type':'integer','minimum':6,'maximum':100},
    max_steps={'type':'integer','minimum':1,'maximum':10000},execute_steps={'type':'integer','minimum':1,'maximum':40},
    capture_every_chunks={'type':'integer','minimum':1,'maximum':100},
    profile_fraction={'type':'number','minimum':.2,'maximum':.7},evaluation_fraction={'type':'number','minimum':.1,'maximum':.4}))


def default_design():return dict(schema_version='1.0',seed=42,states_per_task=30,max_steps=720,execute_steps=8,
    capture_every_chunks=4,profile_fraction=.5,evaluation_fraction=.2)


def make_capture_plan(bank_directories,design=None):
    d=deepcopy(design or default_design());validate(d,DESIGN_SCHEMA)
    if d['profile_fraction']+d['evaluation_fraction']>.9:raise ValueError('At least 10% holdout required')
    rows=[];banks={};all_groups=set();tasks=set()
    for directory in bank_directories:
        directory=Path(directory);m,states=load_state_bank(directory);task=m['task']
        if task in tasks:raise ValueError('Use one initial-state bank per task')
        tasks.add(task);bid=digest(dict(task=task,manifest_sha256=sha256(directory/'initial_states.json')))
        banks[bid]=dict(task=task,manifest_sha256=sha256(directory/'initial_states.json'),states_file_sha256=m['states_file_sha256'])
        if len(m['cases'])<d['states_per_task']:raise ValueError('Not enough saved states for task: '+task)
        cases=sorted(m['cases'],key=lambda x:x['case_id']);random.Random(str(d['seed'])+task).shuffle(cases)
        # Reject duplicate states even outside the selected subset.
        groups=[digest(dict(task=task,state_sha256=c['state_sha256'])) for c in cases]
        if len(set(groups))!=len(groups) or set(groups)&all_groups:raise ValueError('Duplicate task/initial-state group')
        all_groups.update(groups);n=d['states_per_task'];a=max(1,math.floor(n*d['profile_fraction']));b=max(1,math.floor(n*d['evaluation_fraction']))
        if a+b>=n:raise ValueError('Empty holdout split')
        for i,c in enumerate(cases[:n]):
            rows.append(dict(bank_id=bid,case=deepcopy(c),task=task,group_id=digest(dict(task=task,state_sha256=c['state_sha256'])),
                split='profile' if i<a else 'evaluation' if i<a+b else 'holdout'))
    if not rows:raise ValueError('No initial-state banks')
    chunks=math.ceil(d['max_steps']/d['execute_steps']);frames=1+math.ceil(chunks/d['capture_every_chunks'])+1
    p=dict(schema_version='1.0',design=d,banks=banks,cases=rows,limits=dict(episodes=len(rows),action_calls_upper=len(rows)*chunks,
        observation_frames_upper=len(rows)*frames),scope='LIBERO simulation capture; not wall-clock performance measurement')
    p['plan_id']=digest(p);return p


def validate_capture_plan(plan,bank_directories):
    if make_capture_plan(bank_directories,plan['design'])!=plan:raise ValueError('Capture plan/bank changed')
    return plan


def capture(plan,bank_directories,output,env_factory,transport_factory,*,execution_kind):
    validate_capture_plan(plan,bank_directories)
    if execution_kind not in ('cpu_dummy','libero_gpu'):raise ValueError('Unknown capture execution provenance')
    banks={}
    for directory in bank_directories:
        m,states=load_state_bank(directory);bid=digest(dict(task=m['task'],manifest_sha256=sha256(Path(directory)/'initial_states.json')))
        if execution_kind=='libero_gpu' and m['source'].get('kind')!='LIBERO_benchmark_initial_states':raise ValueError('Real capture requires LIBERO state provenance')
        banks[bid]=(m,states)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write_json(out/'capture_plan.json',plan)
    write_json(out/'source_manifest.json',runtime_source_manifest())
    sink=JsonlSink(out/'events.jsonl');outcomes=[];run_id='capture-'+uuid.uuid4().hex;d=plan['design']
    try:
        for index,row in enumerate(plan['cases']):
            m,states=banks[row['bank_id']];case=row['case'];folder=out/f'episode-{index:04d}';folder.mkdir()
            result=dict(group_id=row['group_id'],split=row['split'],task=row['task'],case_id=case['case_id'],status='failed',
                        actual_steps=0,action_chunks=0,observations=0,simulator_success=None)
            env=transport=None
            try:
                env=env_factory(row['task']);obs,restored=env.reset(states[case['case_id']].copy(),case['seed'])
                transport=transport_factory();client=MeasuredClient(run_id=run_id,robot_id=f'capture-{index}',
                    reset_call=transport.reset,predict_call=lambda meta,o:transport.predict(meta,o,case['seed']+meta['observation_seq']),
                    sink=sink,clock_domain=run_id+'-driver-monotonic')
                client.reset(case['case_id']);last_saved=None;success=getattr(env,'last_success',None)
                def save(stage):
                    nonlocal last_saved
                    key=result['action_chunks']
                    if last_saved==key:return
                    save_observation(folder/f'observation-{key:04d}',obs,dict(task=row['task'],case=case,source=m['source'],
                        execution_kind=execution_kind,arm='official_capture',observation_seq=key,executed_steps=result['actual_steps'],
                        restored_state_sha256=restored,capture_plan_id=plan['plan_id'],capture_split=row['split'],
                        capture_stage=stage,simulator_success=success))
                    last_saved=key;result['observations']+=1
                save('initial')
                max_chunks=math.ceil(d['max_steps']/d['execute_steps'])
                while not success and result['actual_steps']<d['max_steps'] and result['action_chunks']<max_chunks:
                    if result['action_chunks']%d['capture_every_chunks']==0:save('progress')
                    q=client.predict(obs)
                    if q is None:raise RuntimeError('Policy output rejected; no action applied')
                    step={}
                    def execute(action):
                        limit=min(d['execute_steps'],d['max_steps']-result['actual_steps'])
                        np.savez_compressed(folder/f"action-{result['action_chunks']:04d}.npz",**action)
                        step.update(env.step(action,limit));n=step['steps']
                        if type(n) is not int or not 1<=n<=limit:raise ValueError('Invalid actual environment step count')
                        return n
                    n=client.apply(q,execute);result['actual_steps']+=n;result['action_chunks']+=1
                    obs=step['observation'];success=bool(step['success'])
                    if success or step['done']:
                        result['termination']='success' if success else 'environment_done';break
                else:result['termination']='success_at_reset' if success else 'step_limit' if result['actual_steps']>=d['max_steps'] else 'chunk_budget'
                save('terminal');result.update(status='completed',simulator_success=success)
            except Exception as exc:result['error']=dict(type=type(exc).__name__,message=str(exc))
            finally:
                for resource in (transport,env):
                    if resource is not None:
                        try:resource.close()
                        except Exception as exc:result.setdefault('cleanup_errors',[]).append(type(exc).__name__)
                outcomes.append(result);write_json(folder/'outcome.json',result)
    finally:
        sink.close()
        summary=dict(plan_id=plan['plan_id'],execution_kind=execution_kind,intended_episodes=len(plan['cases']),
            started_episodes=len(outcomes),completed_episodes=sum(x['status']=='completed' for x in outcomes),
            observations=sum(x['observations'] for x in outcomes),outcomes=outcomes,
            quality_labels_assigned=False,safety_hazard_coverage_guaranteed=False,performance_benchmark=False)
        write_json(out/'capture_summary.json',summary)
    return summary
