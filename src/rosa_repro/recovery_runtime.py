"""Journal a fallback motion separately from the rejected model action."""
import time,uuid
from copy import deepcopy
import numpy as np
from .adapter import ProtocolError
from .paired import digest_value
from .regression import sha256
from .recovery_policy import valid_authorization,candidates
from .hybrid_safety import valid_receipt


async def attempt_retreat(*,sim,plan,state,outcome,parent,template,limit,end,out,life,
                          snapshot,capture_size,visual_blocked,until,action_time_s):
    rid=uuid.uuid4().hex;started=time.monotonic_ns();version=state['observation_seq']-1
    event=dict(kind='recovery_decision',recovery_id=rid,parent_request_id=parent['request_id'],
        robot_id=parent['robot_id'],episode_id=parent['episode_id'],start_ns=started,
        observation_version=version,requested_steps=limit,applied=False,executed_steps=0,
        motion_source='retreat',counts_as_S1_success=False)
    life(dict(kind='recovery_started',**{k:v for k,v in event.items() if k!='kind'}))
    response=await sim.retreat(deepcopy(template),limit)
    event.update(preview_end_ns=time.monotonic_ns(),response={k:v for k,v in response.items() if k!='action'})
    try:
        if response['status']!='clear':event['reason']=response['reason'];return None
        action=response['action'];proof=response['evidence']
        choices=dict(candidates(template,plan['recovery_policy']))
        if response.get('candidate') not in choices or digest_value(action)!=digest_value(choices[response['candidate']]):
            raise ProtocolError('Retreat differs from declared candidate')
        if not valid_authorization(proof,plan['collision_guard'],plan['recovery_policy'],action,limit,'retreat'):
            raise ProtocolError('Retreat authorization invalid')
        if time.monotonic_ns()>=end:event['reason']='cutoff';return None
        if visual_blocked() or state['safety_observation_version']!=version:
            event['reason']='fresh_visual_unavailable';return None
        capture_size(action);path=out/(rid+'-retreat.npz');np.savez_compressed(path,**action)
        begin=time.monotonic_ns();event.update(action_start_ns=begin,visual_request_id=state['safety_request_id'],
            file=path.name,sha256=sha256(path),action_sha256=digest_value(action))
        life(dict(kind='recovery_action_started',recovery_id=rid,start_ns=begin))
        outcome['steps_unknown']=True
        step=await sim.step(deepcopy(action),limit)
        if not valid_receipt(step.get('collision_guard'),proof,step['steps']):raise ProtocolError('Retreat receipt differs')
        outcome['steps_unknown']=False;outcome['actual_steps']+=step['steps'];outcome['action_chunks']+=1
        snapshot(step['observation'])
        event.update(applied=True,executed_steps=step['steps'],receipt=step['collision_guard'],
            recovery_state=step['recovery_state'],new_observation_version=state['observation_seq']-1,reason='applied')
        # Every subsequent response must have seen the post-retreat observation.
        state.update(plan=None,uses=0,safety=None,safety_request_id=None,safety_expiry_ns=0,
            minimum_safety_observation_version=state['observation_seq']-1,
            recovery_feedback='A geometry-checked upward retreat was applied after the previous motion was rejected. '
                'Replan from the current images; avoid repeating the blocked approach.',
            pending_recovery=rid,resume_streak=0)
        await until(begin+round(action_time_s*1e9))
        return step
    finally:
        event['end_ns']=time.monotonic_ns();life(event)
