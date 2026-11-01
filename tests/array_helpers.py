import importlib.util
from pathlib import Path
from rosa_repro.action_regression import collect_arrays
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.dummy import DummyPolicy,observation
from rosa_repro.regression import save_observation

spec=importlib.util.spec_from_file_location('array_capture_script',Path(__file__).resolve().parents[1]/'scripts/r02_compare.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
capture=module.capture

class Client:
    def __init__(self,policy=None,server=None):
        self.policy=policy or DummyPolicy()
        self.server=server or dict(rosa_source_manifest=runtime_source_manifest(),runtime={'observed_torch_state':{'cuda_available':False}},kind='CPU test fixture')
    def call_endpoint(self,name,params=None,**kwargs):
        if name=='r02_runtime':return self.server
        return collect_arrays(self.policy,**params)

def inputs(root,n=4):
    for i in range(n):save_observation(root/str(i),observation(i+1),{'kind':'synthetic_CPU'})
    return root

class StochasticPolicy(DummyPolicy):
    def get_action(self,obs):
        import torch,random,numpy as np
        actions,_=super().get_action(obs)
        actions['action.x']+=torch.rand(actions['action.x'].shape).numpy()+np.random.random()+random.random()
        return actions,{}
