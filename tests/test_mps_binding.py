from copy import deepcopy
import json
import pytest
from test_prototype_validation import configs
from rosa_repro.prototype_plan import prepare,validate
from rosa_repro.execution_plan import validate_binding
from rosa_repro.runtime_attestation import static_environment

ENV={'CUDA_MPS_PIPE_DIRECTORY':'/tmp/rosa-mps/pipes','CUDA_MPS_LOG_DIRECTORY':'/tmp/rosa-mps/logs'}


def test_mps_environment_is_plan_bound_and_observed_without_credentials(monkeypatch):
    cs=configs()
    for c in cs.values():c['binding']['runtime_env']=dict(py_executable='/opt/python',env_vars=deepcopy(ENV))
    p=prepare(cs,{'P4':1});validate(p)
    assert p['workers'][0]['backend']['runtime_env']['env_vars']==ENV
    bad=deepcopy(p);bad['workers'][0]['backend']['runtime_env']['env_vars']['CUDA_MPS_PIPE_DIRECTORY']='/tmp/different'
    with pytest.raises(ValueError):validate(bad)
    for key,value in ENV.items():monkeypatch.setenv(key,value)
    monkeypatch.setenv('HF_TOKEN','do-not-capture');monkeypatch.setenv('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','50')
    e=static_environment()
    assert all(e['selected_environment'][k]==v for k,v in ENV.items())
    assert e['selected_environment']['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']=='50' and 'do-not-capture' not in json.dumps(e)


@pytest.mark.parametrize('value',[{'HF_TOKEN':'secret'},{'CUDA_MPS_PIPE_DIRECTORY':'/tmp/pipe'},
    dict(ENV,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='101'),dict(ENV,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50),
    dict(ENV,CUDA_MPS_PIPE_DIRECTORY='relative'),dict(ENV,CUDA_MPS_LOG_DIRECTORY='/tmp/../other'),
    dict(ENV,PYTHONPATH='/tmp/injected')])
def test_binding_rejects_unapproved_environment_or_invalid_mps_controls(value):
    c=configs()['s1']['binding'];c['runtime_env']=dict(py_executable='/opt/python',env_vars=value)
    with pytest.raises(ValueError):validate_binding(c,'cpu_dummy')


def test_mps_active_thread_limit_is_explicit_and_does_not_change_deadlines():
    cs=configs()
    for c in cs.values():c['binding']['runtime_env']=dict(py_executable='/opt/python',env_vars=dict(ENV,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50'))
    p=prepare(cs,{'P4':1});validate(p)
    assert {m['role']:m['deadline_ms'] for m in p['classes'][0]['audit']['components']}==dict(s1=200,s2=2000,safety=500,monitor=2000)
