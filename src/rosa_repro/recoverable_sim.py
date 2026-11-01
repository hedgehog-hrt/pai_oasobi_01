"""Simulation-only anticipatory guard and bounded, independently checked retreat."""
from copy import deepcopy
import random,time,uuid
import numpy as np
from .hybrid_safety import GuardedLiberoEpisode,GeometryChecker,CollisionDetected,PROTOCOL
from .shadow_checkpoint import ShadowCheckpoint,end_effector_position
from .recovery_policy import validate_config,Limits,candidates
from .paired import digest_value


class RecoverableLiberoEpisode(GuardedLiberoEpisode):
    def __init__(self,*args,recovery_policy,**kwargs):
        self.policy=validate_config(recovery_policy);self.checkpoint=None
        self.limits=Limits(self.policy)
        super().__init__(*args,**kwargs)

    def reset(self,state,seed):
        self.checkpoint=None;self.limits=Limits(self.policy)
        return super().reset(state,seed)

    def synchronize(self):
        if self.checkpoint is None:
            super().synchronize();self.checkpoint=ShadowCheckpoint(self.shadow)
        elif self.dirty:
            self.checkpoint.restore(self.shadow);self.match(self.state(self.live),self.state(self.shadow));self.dirty=False

    def _forecast(self,action,n):
        if type(n) is not int or not 1<=n<=self.live.contract.n_action_steps:raise ValueError('Invalid forecast horizon')
        start=time.monotonic_ns();before=self.state(self.shadow);pos=end_effector_position(self.shadow)
        e=dict(protocol=PROTOCOL,config_sha256=digest_value(self.settings),action_sha256=digest_value(action),
            state_before_sha256=digest_value(before),requested_steps=n,live_steps_executed=0,status='unknown',lease_id=None,recovery_ms=0.)
        sim=self.shadow.base._env.env.sim;original=sim.step;checker=GeometryChecker(sim,self.settings)
        self.dirty=True;after=None;preview=None
        try:
            checker.check()
            def checked_step(*args,**kwargs):
                original(*args,**kwargs);checker.steps+=1;checker.check()
                if (time.monotonic_ns()-start)/1e6>self.settings['max_physics_preview_ms']:
                    raise TimeoutError('Collision preview budget exceeded')
            sim.step=checked_step;preview=self.preview_steps(action,n);after=self.state(self.shadow)
            delta=end_effector_position(self.shadow)-pos
            e.update(status='clear',lease_id=uuid.uuid4().hex,predicted_state_after_sha256=digest_value(after),
                predicted_steps=preview['steps'],predicted_success=preview['success'],predicted_done=preview['done'],
                end_effector_delta_m=delta.tolist(),displacement_m=float(np.linalg.norm(delta)))
        except CollisionDetected as exc:e.update(status='blocked',violation=exc.evidence)
        except Exception as exc:e.update(status='unknown',error_type=type(exc).__name__,error=str(exc))
        finally:
            sim.step=original;e.update(checker.evidence());e['elapsed_ms']=(time.monotonic_ns()-start)/1e6
            e['physics_preview_ms']=e['elapsed_ms'];e['timely']=e['elapsed_ms']<=self.settings['max_physics_preview_ms']
            if not e['timely']:e.update(status='unknown',lease_id=None,error_type='PreviewDeadlineExceeded')
        return e,after,preview

    def _authorize(self,action,n,kind,started):
        self.synchronize();before=self.state(self.live);self.match(before,self.state(self.shadow))
        e,after,preview=self._forecast(action,n)
        e.update(recovery_policy_sha256=digest_value(self.policy),motion_source=kind,
            checkpoint_protocol='libero_t01_native_shadow_checkpoint_v1',checkpoint_sha256=self.checkpoint.identity,
            nominal_geometry_status=e['status'],terminal_witness=None,witness_attempts=[])
        if e['status']=='clear':
            terminal=ShadowCheckpoint(self.shadow)
            if kind=='retreat' and not self.limits.accepts(e['end_effector_delta_m']):
                e.update(status='blocked',lease_id=None,violation={'kind':'retreat_displacement_limit'})
            elif preview['success'] or preview['done']:
                e['terminal_witness']={'kind':'terminal_episode','state_sha256':e['predicted_state_after_sha256']}
            else:
                for name,escape in candidates(action,self.policy):
                    terminal.restore(self.shadow)
                    proof,_,_=self._forecast(escape,n)
                    e['witness_attempts'].append(dict(candidate=name,evidence=proof))
                    if proof['status']=='clear' and self._witness_displacement(proof):
                        e['terminal_witness']=dict(kind='retreat',candidate=name,evidence=proof);break
                    if proof['status']=='unknown':break
                    if (time.monotonic_ns()-started)/1e6>self.policy['max_decision_ms']:break
                if e['terminal_witness'] is None:
                    uncertain=any(x['evidence']['status']=='unknown' for x in e['witness_attempts'])
                    e.update(status='unknown' if uncertain else 'blocked',lease_id=None,
                        violation={'kind':'terminal_retreat_unverified' if uncertain else 'no_terminal_retreat'})
            terminal.restore(self.shadow)
        e['elapsed_ms']=(time.monotonic_ns()-started)/1e6
        e['timely']=bool(e['timely'] and e['elapsed_ms']<=min(self.settings['max_preview_ms'],self.policy['max_decision_ms']))
        if not e['timely']:e.update(status='unknown',lease_id=None,error_type='RecoveryDecisionDeadlineExceeded')
        if e['status']=='clear':
            self.lease=dict(action_sha256=digest_value(action),state_before=before,state_after=after,
                requested_steps=n,preview=preview,evidence=deepcopy(e),motion_source=kind,
                displacement=e['displacement_m'] if kind=='retreat' else 0.)
            self.dirty=True
        else:
            self.checkpoint.restore(self.shadow);self.dirty=False;self.lease=None
        self.match(before,self.state(self.live))
        return e

    def _witness_displacement(self,e):
        d=np.asarray(e['end_effector_delta_m'])
        return bool(np.isfinite(d).all() and d[2]>=self.policy['min_up_displacement_m'] and np.linalg.norm(d)<=self.policy['max_retreat_displacement_m'])

    def guard(self,action,requested_steps):
        self.lease=None;started=time.monotonic_ns();rng=(np.random.get_state(),random.getstate())
        try:return self._authorize(action,requested_steps,'s1',started)
        finally:np.random.set_state(rng[0]);random.setstate(rng[1])

    def retreat(self,template,requested_steps):
        self.lease=None;started=time.monotonic_ns();rng=(np.random.get_state(),random.getstate())
        result=dict(status='blocked',reason='retreat_limits',action=None,evidence=None,candidates=[],limits=self.limits.evidence())
        try:
            if not self.limits.available():return result
            for name,action in candidates(template,self.policy):
                e=self._authorize(action,requested_steps,'retreat',started)
                result['candidates'].append(dict(candidate=name,evidence=e))
                if e['status']=='clear':
                    result.update(status='clear',reason='verified_retreat',candidate=name,action=action,evidence=e);return result
                if e['status']=='unknown':
                    result.update(status='unknown',reason='retreat_verification_unknown');return result
            result['reason']='no_verified_retreat';return result
        finally:np.random.set_state(rng[0]);random.setstate(rng[1])

    def step(self,action,requested_steps):
        if self.lease is None:raise ValueError('No recovery-aware lease')
        source=self.lease['motion_source'];distance=self.lease['displacement']
        result=super().step(action,requested_steps)
        self.limits.applied(source,distance)
        self.checkpoint=ShadowCheckpoint(self.shadow)
        result['recovery_state']=dict(policy_sha256=digest_value(self.policy),motion_source=source,**self.limits.evidence())
        return result
