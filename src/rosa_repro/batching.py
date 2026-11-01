"""Batch assembly, session admission and output splitting, independent of Ray."""
from dataclasses import asdict
import time
import uuid

import numpy as np
from .adapter import Request,ProtocolError,check_observation_batch_one,check_action_batch_one


def observation_signature(obs):
    check_observation_batch_one(obs)
    return tuple((k,str(v.dtype),v.shape[1:]) if isinstance(v,np.ndarray) and k.startswith(('video.','state.'))
                 else (k,'language') for k,v in sorted(obs.items()))


def combine_observations(observations):
    signatures=[observation_signature(o) for o in observations]
    if not signatures or any(s!=signatures[0] for s in signatures):
        raise ProtocolError('Incompatible observation batch')
    result={}
    for key in observations[0]:
        if key.startswith(('video.','state.')):
            result[key]=np.concatenate([o[key] for o in observations],axis=0)
        else:
            result[key]=[o[key] if isinstance(o[key],str) else o[key][0] for o in observations]
    return result


def split_actions(action,count):
    if not isinstance(action,dict) or not action:
        raise ProtocolError('Empty batched action')
    if any(not isinstance(v,np.ndarray) or v.ndim!=3 or v.shape[0]!=count for v in action.values()):
        raise ProtocolError('Model output batch dimension mismatch')
    split=[{k:v[i:i+1].copy() for k,v in action.items()} for i in range(count)]
    for item in split: check_action_batch_one(item)
    return split


def predict_policy_batch(policy,observations):
    """The exact shared GR00T tensor path, also exercised by saved-array regression."""
    action,_=policy.get_action(combine_observations(observations))
    return split_actions(action,len(observations))


class SessionBook:
    """Single event-loop ownership. No awaits inside admission/reset/finish."""
    def __init__(self):
        self.sessions={}; self.episodes=set(); self.requests=set(); self.signature=None
    def reset(self,run_id,robot_id,episode_id):
        Request(run_id,robot_id,episode_id,'reset','reset','s1',0,0)
        key=(run_id,robot_id)
        if (*key,episode_id) in self.episodes: raise ProtocolError('Duplicate episode ID')
        previous=self.sessions.get(key)
        generation=0 if previous is None else previous['generation']+1
        self.sessions[key]=dict(episode_id=episode_id,generation=generation,last_seq=-1,busy=False)
        self.episodes.add((*key,episode_id))
        return dict(run_id=run_id,robot_id=robot_id,episode_id=episode_id,generation=generation)
    def accept(self,metadata,obs):
        req=Request(**metadata)
        signature=observation_signature(obs)
        state=self.sessions.get((req.run_id,req.robot_id))
        if state is None or state['episode_id']!=req.episode_id or state['generation']!=req.generation:
            raise ProtocolError('Stale/unknown episode')
        if (req.run_id,req.request_id) in self.requests: raise ProtocolError('Duplicate request ID')
        if state['busy'] or req.observation_seq<=state['last_seq']: raise ProtocolError('Concurrent/out-of-order request')
        if self.signature is not None and self.signature!=signature: raise ProtocolError('Incompatible observation schema')
        self.signature=signature
        self.requests.add((req.run_id,req.request_id));state['busy']=True;state['last_seq']=req.observation_seq
        return dict(request=req,state=state,enqueued_ns=time.monotonic_ns())
    def current(self,ticket):
        r=ticket['request'];return self.sessions[(r.run_id,r.robot_id)] is ticket['state']
    def finish(self,ticket): ticket['state']['busy']=False


def empty_response(ticket,status='cancelled'):
    return dict(metadata=asdict(ticket['request']),status=status,action=None,batch_id=None,actual_batch_size=None)
