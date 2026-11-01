"""Lazy imports for pinned LIBERO/GR00T; no simulator dependency for CPU tests."""
from pathlib import Path

import numpy as np

from .paired import digest_value, save_state_bank
from .regression import UPSTREAM_COMMIT

T01_NAME='pick_up_the_alphabet_soup_and_place_it_in_the_basket'
LIBERO_COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'


def _load_initial_states(path):
    """Keep PyTorch's restricted loader while allowing the pinned NumPy states."""
    import torch
    from numpy.core.multiarray import _reconstruct
    # LIBERO serializes NumPy 1.x names; also support the canonical name when
    # running our CPU regression checks under NumPy 2.x.
    allowed=[np.ndarray,np.dtype,_reconstruct,
             (_reconstruct,'numpy.core.multiarray._reconstruct'),type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        states=torch.load(path,weights_only=True)
    if (not isinstance(states,np.ndarray) or states.dtype!=np.float64 or
            states.ndim!=2 or min(states.shape)<1 or not np.isfinite(states).all()):
        raise ValueError('Expected finite float64 LIBERO initial-state matrix')
    return states


def task_and_states(task_name=T01_NAME):
    from libero.libero import benchmark
    import inspect
    import subprocess
    directory=Path(inspect.getfile(benchmark)).resolve().parent
    actual=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if actual!=LIBERO_COMMIT:
        raise ValueError('LIBERO revision differs from the pinned baseline')
    suite=benchmark.get_benchmark_dict()['libero_object']()
    matches=[i for i in range(suite.get_num_tasks()) if suite.get_task(i).name==task_name]
    if len(matches)!=1:
        raise ValueError('Pinned LIBERO task not uniquely found')
    task_id=matches[0]
    task=suite.get_task(task_id)
    from libero.libero.utils import get_libero_path
    path=(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file).resolve()
    root=Path(subprocess.check_output(['git','-C',str(directory),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
    relative=path.relative_to(root)
    committed=subprocess.check_output(['git','-C',str(root),'show',f'{LIBERO_COMMIT}:{relative.as_posix()}'])
    if path.read_bytes()!=committed:
        raise ValueError('LIBERO initial-state file differs from pinned commit')
    return task,_load_initial_states(path),task_id


def export_t01_bank(output, count=10, seed=42,task_name=T01_NAME):
    task,states,task_id=task_and_states(task_name)
    if not 1<=count<=len(states):
        raise ValueError('Initial-state count exceeds available LIBERO states')
    return save_state_bank(output,[np.asarray(s) for s in states[:count]],task=task_name,
        source=dict(kind='LIBERO_benchmark_initial_states',suite='libero_object',task_id=task_id,
                    gr00t_commit=UPSTREAM_COMMIT,libero_commit=LIBERO_COMMIT),seed=seed)


class LiberoEpisode:
    def __init__(self, modality_config, max_steps=720, execute_steps=8,task_name=T01_NAME):
        from gr00t.eval._horizon_contract import PolicyHorizonSpec
        from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
        from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
        from libero.libero.utils import get_libero_path
        task,_,_=task_and_states(task_name)
        class FixedStateEnv(LiberoEnv):
            def reset(self, seed=None, options=None):
                super().reset(seed=seed, options=options)
                obs=self._env.set_init_state(options['state'])
                self.restored_hash=digest_value(self._env.get_sim_state())
                return self._process_observation(obs),{'success':self._env.check_success()}
        self.base=FixedStateEnv(str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),task.language)
        self.contract=PolicyHorizonSpec.from_modality_config(modality_config,n_action_steps=execute_steps)
        self.env=MultiStepWrapper(self.base,contract=self.contract,max_episode_steps=max_steps,terminate_on_success=True)

    def _batch(self, obs):
        # Use the same Gymnasium space-aware batching/casts as vector eval.
        from gymnasium.vector.utils import concatenate,create_empty_array
        return concatenate(self.env.observation_space,[obs],create_empty_array(self.env.observation_space,n=1))

    def reset(self,state,seed):
        obs,_=self.env.reset(seed=seed,options={'state':state})
        self.last_success=bool(self.base._env.check_success())
        return self._batch(obs),self.base.restored_hash

    def step(self,action,requested_steps):
        if not 1<=requested_steps<=self.contract.n_action_steps:
            raise ValueError('Invalid execution stride')
        self.env.n_action_steps=requested_steps
        obs,_,terminated,truncated,info=self.env.step({k:v[0] for k,v in action.items()})
        return dict(observation=self._batch(obs),steps=int(info['n_env_steps']),
                    success=bool(np.any(info['success'])),done=bool(terminated or truncated))

    def close(self):
        self.env.close()


class ZmqRegressionTransport:
    def __init__(self,arm,port):
        from gr00t.policy.server_client import PolicyClient
        self.arm=arm
        self.client=PolicyClient(host='127.0.0.1',port=port,timeout_ms=120000)

    def reset(self,run_id,robot_id,episode_id):
        return self.client.call_endpoint('r02_reset',dict(run_id=run_id,robot_id=robot_id,episode_id=episode_id))

    def predict(self,metadata,observation,seed):
        import zmq
        try:
            return self.client.call_endpoint('r02_predict_seeded',dict(
                metadata=metadata,observation=observation,seed=seed,arm=self.arm))
        except zmq.error.Again as exc:
            raise TimeoutError('Regression server timeout') from exc

    def close(self):
        self.client.close()
