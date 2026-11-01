"""Rejoin dynamic control logs to saved observations, semantic plans and physical actions."""
import json
from pathlib import Path
import numpy as np
from .global_audit import verify_tree,evaluate
from .regression import load_observation,sha256
from .paired import digest_value
from .profile_adoption import inside,read
from .component_runtime import validate_output
from .execution_plan import runtime_source_manifest


def reaudit(root):
    p=Path(root);verify_tree(p);result=evaluate(p/'control');sim=p/'simulation'
    evidence=read(sim/'simulators.json');mode=result['mode'];kind='cpu_dummy' if mode=='cpu_dummy' else 'libero'
    if any(e['source_manifest']!=runtime_source_manifest() or e['kind']!=kind for e in evidence.values()):raise ValueError('Simulator source/kind changed')
    rows=[json.loads(x) for x in (p/'control/journal.jsonl').read_text().splitlines()];starts={};ends={};actions={};finished={};epochs={r['directory']:r['epoch'] for r in rows if r['kind']=='generation_ready'}
    for directory,epoch in epochs.items():
        for e in (json.loads(x) for x in (p/'control'/directory/'events.jsonl').read_text().splitlines()):
            prefix=str(epoch)+':'
            if e['kind']=='prediction_started':starts[prefix+e['metadata']['request_id']]=e
            elif e['kind'] in ('prediction_finished','prediction_failed'):ends[prefix+e['request_id']]=e
            elif e['kind']=='action_started':actions[prefix+e['ticket_id']]=e
            elif e['kind']=='action_finished':finished[prefix+e['ticket_id']]=e
    requests=[json.loads(x) for x in (sim/'request_observations.jsonl').read_text().splitlines()];seen=set()
    for r in requests:
        q=r['request_id']
        if q in seen or q not in starts:raise ValueError('Duplicate or unknown request capture')
        seen.add(q);s=starts[q];obs,manifest=load_observation(inside(sim,r['observation_directory']))
        e=ends[q];expected=e.get('status') if e['kind']=='prediction_finished' else 'timeout' if e['error_type']=='TimeoutError' else 'error'
        if r['status']!=expected or r['component_id']!=s['metadata']['component_id'] or not r['invoked_ns']<=s['time_ns']<=e['time_ns']<=r['returned_ns']:raise ValueError('Driver request status/timing differs from control events')
        if r['robot_id']!=s['robot_id'] or manifest['provenance']['robot_id']!=s['robot_id']:raise ValueError('Observation robot identity differs')
        pid=s.get('plan_request_id');prefix=q.split(':',1)[0]+':'
        plan=ends[prefix+pid]['semantic_output']['plan'] if pid else None
        if digest_value(dict(observation=obs,plan=plan,context=r['context']))!=s['input_sha256']:raise ValueError('Dynamic observation/plan input hash differs')
    if result['status']=='complete' and seen!=set(starts):raise ValueError('Missing dynamic observation capture')
    rows=[json.loads(x) for x in (sim/'actions.jsonl').read_text().splitlines()];begins={};steps={}
    for e in rows:
        target=begins if e['kind']=='started' else steps if e['kind']=='stepped' else None
        if target is None or e['ticket_id'] in target:raise ValueError('Duplicate/unknown simulator action event')
        target[e['ticket_id']]=e
    if set(begins)!=set(actions):raise ValueError('Physical action intent coverage differs')
    count=0
    for q,e in begins.items():
        path=inside(sim,e['action_file'])
        with np.load(path,allow_pickle=False) as data:action={k:data[k] for k in data.files}
        validate_output('s1',action)
        if sha256(path)!=e['action_sha256'] or digest_value(action)!=e['input_sha256'] or e['input_sha256']!=actions[q]['output_sha256'] or e['input_sha256']!=ends[q]['output_sha256']:raise ValueError('Applied action differs from S1 output')
        f=finished[q]
        if f['status']=='completed':
            step=steps[q]
            if step['executed_steps']!=f['steps'] or not actions[q]['time_ns']<=e['time_ns']<=step['time_ns']<=f['time_ns'] or step['after_version']!=e['before_version']+1:raise ValueError('Physical steps/version/time differ')
            count+=1
    cleanup_ok=read(sim/'cleanup.json')['status']=='complete'
    if (p/'cleanup.json').exists():cleanup_ok=cleanup_ok and read(p/'cleanup.json').get('own_fleet_cleanup_confirmed') is True
    return dict(status=result['status'] if cleanup_ok else 'incomplete',cleanup_confirmed=cleanup_ok,dynamic_evaluation=result,observations_revalidated=len(seen),actions_revalidated=count,
        source_manifest=runtime_source_manifest(),physical_steps_semantics='simulator-reported, action tensor and control log revalidated',GPU_performance_claim=False)
