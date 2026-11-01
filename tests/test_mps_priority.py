from copy import deepcopy
import os
import pytest
from jsonschema.exceptions import ValidationError
from rosa_repro.mps_priority import attest,engine_launch
from rosa_repro.vlm_contract import default_config,runtime_fingerprint,validate_config
from rosa_repro.vlm_backends import vllm_engine_options
from rosa_repro.execution_plan import validate_binding


def test_child_launch_scope_restores_parent_even_on_failure(monkeypatch):
    monkeypatch.setenv('CUDA_MPS_PIPE_DIRECTORY','/tmp/test-pipes')
    monkeypatch.setenv('CUDA_MPS_LOG_DIRECTORY','/tmp/test-logs')
    monkeypatch.setenv('CUDA_MPS_CLIENT_PRIORITY','0')
    with pytest.raises(RuntimeError):
        with engine_launch(dict(priority=1,role='s2')) as evidence:
            assert os.environ['CUDA_MPS_CLIENT_PRIORITY']=='1' and evidence['priority']==1
            assert not evidence['scheduling_guaranteed']
            raise RuntimeError('engine launch failed')
    assert os.environ['CUDA_MPS_CLIENT_PRIORITY']=='0'
    monkeypatch.delenv('CUDA_MPS_CLIENT_PRIORITY')
    with engine_launch(dict(priority=1,role='monitor')):pass
    assert 'CUDA_MPS_CLIENT_PRIORITY' not in os.environ


def test_worker_checks_actual_environment_not_only_configuration():
    env=dict(CUDA_MPS_PIPE_DIRECTORY='/tmp/pipes',CUDA_MPS_LOG_DIRECTORY='/tmp/logs',CUDA_MPS_CLIENT_PRIORITY='1')
    assert attest(dict(priority=1,role='s2'),env)['priority']==1
    for change in [dict(CUDA_MPS_CLIENT_PRIORITY='0'),dict(CUDA_MPS_PIPE_DIRECTORY='')]:
        with pytest.raises(ValueError):attest(dict(priority=1,role='s2'),dict(env,**change))


def test_priority_changes_fingerprint_and_reaches_gpu_worker():
    config=default_config('s2','vllm');config['vision_attention_backend']='TORCH_SDPA'
    original=runtime_fingerprint(config);config['mps_client_priority']=1
    assert runtime_fingerprint(config)!=original
    options=vllm_engine_options(config,'/tmp/model')
    assert options['additional_config']==dict(rosa_mps_priority=dict(priority=1,role='s2'))
    for value in [-1,2,'1',True]:
        bad=dict(config,mps_client_priority=value)
        with pytest.raises((ValidationError,ValueError)):validate_config(bad)
    with pytest.raises(ValueError):validate_config(dict(config,vision_attention_backend='auto'))


def test_runtime_priority_binding_is_strict():
    b=dict(factory='model:factory',kwargs={},runtime_env=dict(py_executable='/opt/python',env_vars=dict(CUDA_MPS_PIPE_DIRECTORY='/tmp/pipes',CUDA_MPS_LOG_DIRECTORY='/tmp/logs',CUDA_MPS_CLIENT_PRIORITY='0')))
    validate_binding(b,'gpu')
    for value in [0,True,'2','']:
        bad=deepcopy(b);bad['runtime_env']['env_vars']['CUDA_MPS_CLIENT_PRIORITY']=value
        with pytest.raises(ValueError):validate_binding(bad,'gpu')
