"""Explicit bounded simulation retreat policy; never a replacement S1 output."""
import numpy as np
from .component_runtime import validate_output

PROTOCOL='libero_t01_retreat_replan_v1'


def config():
    return dict(protocol=PROTOCOL,simulation_only=True,candidate_up_commands=[.1,.25,.5],
        max_retreat_displacement_m=.08,max_episode_retreat_distance_m=.24,
        min_up_displacement_m=.002,max_retreats_per_episode=6,max_consecutive_retreats=2,
        resumed_s1_chunks_required=3,max_decision_ms=2500.,
        gripper_command=.5,require_terminal_retreat_witness=True,
        require_fresh_visual_after_retreat=True,require_new_P4_plan_after_retreat=True,
        counts_as_S1_success=False)


def validate_config(value):
    if value!=config():raise ValueError('Recovery policy differs from the bounded fixed contract')
    return value


def candidates(template,settings):
    validate_config(settings);validate_output('s1',template)
    if set(template)!={'action.'+k for k in ('x','y','z','roll','pitch','yaw','gripper')}:
        raise ValueError('Retreat requires the pinned seven-channel action space')
    result=[]
    for z in settings['candidate_up_commands']:
        action={k:np.zeros_like(v) for k,v in template.items()}
        action['action.z'][:]=z
        # RLDS 0.5 -> binary raw control 0: keep the existing gripper target.
        action['action.gripper'][:]=settings['gripper_command']
        result.append((f'up_{z:g}',action))
    return result


class Limits:
    def __init__(self,settings):
        self.settings=validate_config(settings);self.total=0;self.consecutive=0;self.s1_streak=0;self.distance=0.
    def available(self):
        return self.total<self.settings['max_retreats_per_episode'] and self.consecutive<self.settings['max_consecutive_retreats'] and self.distance<self.settings['max_episode_retreat_distance_m']
    def accepts(self,delta):
        delta=np.asarray(delta)
        return (delta.shape==(3,) and np.isfinite(delta).all() and delta[2]>=self.settings['min_up_displacement_m'] and
            np.linalg.norm(delta)<=self.settings['max_retreat_displacement_m'] and
            self.distance+np.linalg.norm(delta)<=self.settings['max_episode_retreat_distance_m'])
    def applied(self,kind,displacement=0.):
        if kind=='s1':
            self.s1_streak+=1
            if self.s1_streak>=self.settings['resumed_s1_chunks_required']:self.consecutive=0
        elif kind=='retreat':
            if not self.available() or not 0<displacement<=self.settings['max_retreat_displacement_m'] or self.distance+displacement>self.settings['max_episode_retreat_distance_m']:
                raise ValueError('Retreat limit exceeded')
            self.total+=1;self.consecutive+=1;self.s1_streak=0;self.distance+=displacement
        else:raise ValueError('Unknown motion source')
    def evidence(self):
        return dict(retreats=self.total,consecutive=self.consecutive,s1_streak=self.s1_streak,total_retreat_distance_m=self.distance)


def valid_authorization(e,guard,policy,action,steps,source):
    from .hybrid_safety import valid_preview
    from .paired import digest_value
    validate_config(policy)
    if source not in ('s1','retreat'):return False
    if not valid_preview(e,guard,digest_value(action),steps):return False
    if e.get('recovery_policy_sha256')!=digest_value(policy) or e.get('motion_source')!=source:return False
    if e.get('checkpoint_protocol')!='libero_t01_native_shadow_checkpoint_v1' or len(e.get('checkpoint_sha256',''))!=64:return False
    if e['elapsed_ms']>policy['max_decision_ms'] or e.get('nominal_geometry_status')!='clear':return False
    if source=='retreat':
        delta=np.asarray(e.get('end_effector_delta_m',[]))
        if not Limits(policy).accepts(delta) or not np.isclose(np.linalg.norm(delta),e.get('displacement_m',float('nan')),rtol=0,atol=1e-12):return False
    witness=e.get('terminal_witness') or {}
    if witness.get('kind')=='terminal_episode':
        return bool((e.get('predicted_success') or e.get('predicted_done')) and witness.get('state_sha256')==e['predicted_state_after_sha256'])
    if witness.get('kind')!='retreat':return False
    choices=dict(candidates(action,policy));proof=witness.get('evidence');name=witness.get('candidate')
    if name not in choices or not valid_preview(proof,guard,digest_value(choices[name]),steps):return False
    delta=np.asarray(proof.get('end_effector_delta_m',[]))
    return bool(proof['state_before_sha256']==e['predicted_state_after_sha256'] and delta.shape==(3,) and
        np.isfinite(delta).all() and delta[2]>=policy['min_up_displacement_m'] and np.linalg.norm(delta)<=policy['max_retreat_displacement_m'])
