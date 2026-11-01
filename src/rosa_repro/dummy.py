"""Clearly synthetic CPU fixtures; never represent GR00T or LIBERO results."""
import numpy as np
from .adapter import BatchOneWorker
from .paired import digest_value


def observation(value):
    return {'video.image':np.zeros((1,1,2,2,3),np.uint8),
            'state.x':np.array([[[value]]],dtype=np.float32),'task':['CPU fixture']}


class DummyPolicy:
    def get_action(self,obs):
        return {'action.x':np.repeat(obs['state.x'].astype(np.float32),8,axis=1)},{}


class DummyEpisode:
    def reset(self,state,seed):
        self.value=float(state[0]); self.steps=0
        return observation(self.value),digest_value(state)
    def step(self,action,requested_steps):
        if not np.all(action['action.x']==np.float32(self.value)):
            raise ValueError('Response belongs to another fixture')
        n=min(requested_steps,11-self.steps); self.steps+=n
        return dict(observation=observation(self.value),steps=n,success=self.steps==11,done=self.steps==11)
    def close(self): pass


class DummyTransport:
    def __init__(self,arm):
        self.worker=BatchOneWorker(DummyPolicy(),stateless_verified=True)
    def reset(self,*args): return self.worker.reset(*args)
    def predict(self,metadata,obs,seed): return self.worker.predict(metadata,obs)
    def close(self): pass
