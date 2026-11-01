from enum import Enum
from types import SimpleNamespace
import pytest
from rosa_repro.vision_compat import honor_qwen_sdpa
from rosa_repro.vlm_contract import default_config,runtime_fingerprint,validate_config
from rosa_repro.vlm_backends import vllm_engine_options

class Backend(Enum):
    TORCH_SDPA=1
    FLASH_ATTN=2

def test_explicit_choice_survives_upstream_override_and_restores_on_failure():
    original=lambda *a,**kw:(Backend.FLASH_ATTN,object())
    module=SimpleNamespace(maybe_get_vit_flash_attn_backend=original)
    with pytest.raises(RuntimeError,match='load failure'):
        with honor_qwen_sdpa(module,Backend.TORCH_SDPA,vllm_version='0.11.2'):
            assert module.maybe_get_vit_flash_attn_backend(Backend.TORCH_SDPA,False,Backend.TORCH_SDPA)==(Backend.TORCH_SDPA,None)
            with pytest.raises(ValueError):module.maybe_get_vit_flash_attn_backend(Backend.FLASH_ATTN,False,Backend.TORCH_SDPA)
            raise RuntimeError('load failure')
    assert module.maybe_get_vit_flash_attn_backend is original
    with pytest.raises(ValueError):
        with honor_qwen_sdpa(module,Backend.TORCH_SDPA,vllm_version='0.12.0'):pass

def test_override_is_explicit_fingerprinted_and_not_a_transformers_option():
    c=default_config('safety','vllm');before=runtime_fingerprint(c)
    assert 'worker_cls' not in vllm_engine_options(c,'/model')
    c['vision_attention_backend']='TORCH_SDPA';options=vllm_engine_options(c,'/model')
    assert options['mm_encoder_attn_backend']=='TORCH_SDPA'
    assert options['worker_cls']=='rosa_repro.vllm_worker.QwenSdpaWorker' and runtime_fingerprint(c)!=before
    other=default_config('safety','transformers');other['vision_attention_backend']='TORCH_SDPA'
    with pytest.raises(ValueError):validate_config(other)
