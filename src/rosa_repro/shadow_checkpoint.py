"""In-memory branch checkpoint for the pinned LIBERO T01 shadow simulator.

Copy native MjData (including solver caches), controller, gripper, buffers and
observable state. Never transplant this checkpoint into the live environment.
"""
from copy import copy,deepcopy
import random
import numpy as np
from .paired import digest_value

PROTOCOL='libero_t01_native_shadow_checkpoint_v1'


def canonical(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    return value


class ShadowCheckpoint:
    def __init__(self,episode):
        env=episode.base._env.env
        if len(env.robots)!=1 or env.tracking_object_states_change:
            raise ValueError('Checkpoint is restricted to the pinned single-arm static-object task')
        robot=env.robots[0];controller=robot.controller
        if type(controller).__name__!='OperationalSpaceController' or controller.interpolator_pos is not None or controller.interpolator_ori is not None:
            raise ValueError('Unsupported controller checkpoint')
        self.owner=episode;self.sim=env.sim
        self.data=copy(env.sim.data._data)
        self.env={k:deepcopy(getattr(env,k)) for k in ('cur_time','timestep','done','_obs_cache')}
        self.controller=deepcopy({k:v for k,v in vars(controller).items() if k!='sim'})
        self.robot={k:deepcopy(v) for k,v in vars(robot).items() if k.startswith('recent_') or k=='torques'}
        self.gripper=robot.gripper.current_action.copy()
        self.observables={k:deepcopy(vars(v)) for k,v in env._observables.items()}
        self.numpy_rng=np.random.get_state();self.python_rng=random.getstate()
        self.state=np.asarray(episode.base._env.get_sim_state()).copy()
        self.identity=digest_value(canonical(dict(state=self.state,controller=self.controller,gripper=self.gripper,
            timestep=self.env['timestep'],cur_time=self.env['cur_time'],done=self.env['done'])))

    def restore(self,episode):
        if episode is not self.owner or episode.base._env.env.sim is not self.sim:
            raise ValueError('Checkpoint cannot cross simulator instances')
        env=episode.base._env.env;robot=env.robots[0]
        # All wrapper properties delegate to _data at access time. Controller
        # arrays are owned copies; no native data view is retained across restore.
        self.sim.data._data=copy(self.data)
        for k,v in self.env.items():setattr(env,k,deepcopy(v))
        robot.controller.__dict__.clear();robot.controller.__dict__.update(deepcopy(self.controller),sim=self.sim)
        for k,v in self.robot.items():setattr(robot,k,deepcopy(v))
        robot.gripper.current_action=self.gripper.copy()
        if set(env._observables)!=set(self.observables):raise ValueError('Observable set changed')
        for k,v in self.observables.items():
            env._observables[k].__dict__.clear();env._observables[k].__dict__.update(deepcopy(v))
        np.random.set_state(self.numpy_rng);random.setstate(self.python_rng)
        if not np.array_equal(np.asarray(episode.base._env.get_sim_state()),self.state):
            raise ValueError('Native shadow checkpoint did not restore the exact state')


def end_effector_position(episode):
    """Refresh only an independent native data copy, never the forecast caches."""
    import mujoco
    env=episode.base._env.env;data=copy(env.sim.data._data)
    mujoco.mj_fwdPosition(env.sim.model._model,data)
    return np.asarray(data.site_xpos[env.robots[0].eef_site_id]).copy()
