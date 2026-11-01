"""Qwen3 Safety alternative keeps inputs/stop semantics and explicit runtime identity."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from rosa_repro.vlm_contract import default_config,pins,validate_config,runtime_fingerprint,prepare_request,parse_generation
from rosa_repro.vlm_backends import vllm_engine_options
from rosa_repro.vision_compat import honor_qwen3_sdpa
from test_vlm_backends import payload


def candidate(size='4B'):
    c=default_config('safety','vllm');c['model_id']='Qwen/Qwen3-VL-'+size+'-Instruct'
    c['revision']=pins()[c['model_id']]['revision']
    return c


@pytest.mark.parametrize('size',['4B','8B'])
def test_only_model_identity_changes_not_payload_or_stop_rules(size):
    old=default_config('safety','vllm');new=candidate(size);validate_config(new)
    p=payload();a=prepare_request(p,old)[2];b=prepare_request(p,new)[2]
    assert a['images']==b['images'] and a['prompt_sha256']==b['prompt_sha256'] and a['input_sha256']==b['input_sha256']
    assert runtime_fingerprint(new).startswith('qwen3vl-v1-')
    assert runtime_fingerprint(old).startswith('qwen25vl-v1-')
    for label,stop in [('safe',False),('unsafe',True),('unknown',True)]:
        assert parse_generation('safety','{"decision":"'+label+'"}','stop','safety_label_v1')['stop']==stop


def test_alternative_is_explicit_and_revision_bound():
    for key,value in [('role','monitor'),('revision','0'*40),('model_id','Qwen/Qwen3-VL-2B-Instruct')]:
        c=candidate();c[key]=value
        with pytest.raises(ValueError):validate_config(c)
    for role in ['s2','monitor','safety']:
        assert 'Qwen2.5' in default_config(role)['model_id']


def test_qwen3_explicit_sdpa_restored_even_after_load_failure():
    check=lambda dtype:True
    module=SimpleNamespace(check_upstream_fa_availability=check)
    select=lambda *args,**kwargs:('FLASH_ATTN',None)
    attention=SimpleNamespace(maybe_get_vit_flash_attn_backend=select)
    requested=SimpleNamespace(name='TORCH_SDPA')
    with pytest.raises(RuntimeError):
        with honor_qwen3_sdpa(module,requested,vllm_version='0.11.2',attention_module=attention):
            assert module.check_upstream_fa_availability('float16') is False
            assert attention.maybe_get_vit_flash_attn_backend(requested,False)==(requested,None)
            with pytest.raises(ValueError):attention.maybe_get_vit_flash_attn_backend('FLASH_ATTN',False)
            raise RuntimeError('model load failed')
    assert module.check_upstream_fa_availability is check
    assert attention.maybe_get_vit_flash_attn_backend is select
    with pytest.raises(ValueError):
        with honor_qwen3_sdpa(module,SimpleNamespace(name='FLASH_ATTN'),vllm_version='0.11.2',attention_module=attention):pass
    c=candidate();c['vision_attention_backend']='TORCH_SDPA'
    options=vllm_engine_options(c,'/pinned/qwen3')
    assert options['mm_encoder_attn_backend']=='TORCH_SDPA'
    assert options['worker_cls']=='rosa_repro.vllm_worker.QwenSdpaWorker'
