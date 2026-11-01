"""Optional real-model factories. Importing this file does not load GPU models."""
from pathlib import Path
import os
import sys
from .component_runtime import s1_observation
from .batching import predict_policy_batch


class Gr00tBackend:
    def __init__(self,worker,repo,model_path,instruction_key,stateless_verified=False):
        if worker['role']!='s1' or not stateless_verified:raise ValueError('Requires explicitly verified stateless S1 backend')
        from .regression import verify_upstream,runtime_evidence,MODEL_REVISION
        from .checkpoint_verification import verify_groot_checkpoint
        if worker['model_revision']!=MODEL_REVISION:raise ValueError('Worker revision differs from pinned GR00T checkpoint')
        self.worker=worker;self.instruction_key=instruction_key
        repo=Path(repo).resolve();model_path=Path(model_path).resolve()
        self.source=verify_upstream(repo);self.checkpoint=verify_groot_checkpoint(model_path)
        sys.path.insert(0,str(repo));os.chdir(repo)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Expected exactly one Ray-visible CUDA device')
        from gr00t.policy.gr00t_policy import Gr00tPolicy,Gr00tSimPolicyWrapper
        self.policy=Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag='LIBERO_PANDA',model_path=str(model_path),device='cuda:0',strict=True))
        self.runtime=runtime_evidence(self.policy)
    def describe(self):
        return dict(execution_kind='external_model',stateless=True,component_id=self.worker['component_id'],role='s1',
            model_revision=self.worker['model_revision'],runtime_fingerprint=self.worker['runtime_fingerprint'],
            runtime_fingerprint_verification='declared binding; compare observed runtime before formal profiling',
            model_artifacts_verified=self.checkpoint['full_weight_hashes_verified'],
            model_artifact_scope='pinned GR00T inference checkpoint; external runtime dependencies separate',
            device_type='cuda',source=self.source,checkpoint=self.checkpoint,runtime=self.runtime)
    def predict_batch(self,payloads):
        return predict_policy_batch(self.policy,[s1_observation(p,self.instruction_key) for p in payloads])


def groot_backend(worker,**kwargs):return Gr00tBackend(worker,**kwargs)
