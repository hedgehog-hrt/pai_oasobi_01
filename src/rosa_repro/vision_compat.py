"""Scoped fix for vLLM 0.11.2 overriding an explicit Qwen vision SDPA choice.

Does not edit installed upstream files or change decoder attention. The context
is used only while constructing a model inside its dedicated GPU worker.
"""
from contextlib import contextmanager


@contextmanager
def honor_qwen_sdpa(module,requested,*,vllm_version,inherited_override=False):
    if vllm_version!='0.11.2' or getattr(requested,'name',None)!='TORCH_SDPA':
        raise ValueError('Compatibility worker requires pinned vLLM and explicit TORCH_SDPA')
    original=module.maybe_get_vit_flash_attn_backend
    def select(backend,use_upstream_fa,attn_backend_override=None):
        if backend!=requested or (attn_backend_override!=requested and not (inherited_override and attn_backend_override is None)):
            raise ValueError('Model construction changed the declared vision backend')
        return requested,None
    module.maybe_get_vit_flash_attn_backend=select
    try:yield
    finally:module.maybe_get_vit_flash_attn_backend=original


@contextmanager
def honor_qwen3_sdpa(module,requested,*,vllm_version,attention_module):
    """Prevent Qwen3's separate upstream-FA availability branch overriding SDPA."""
    if vllm_version!='0.11.2' or getattr(requested,'name',None)!='TORCH_SDPA':
        raise ValueError('Compatibility worker requires pinned vLLM and explicit TORCH_SDPA')
    original=module.check_upstream_fa_availability
    module.check_upstream_fa_availability=lambda dtype:False
    try:
        # Qwen3 reuses Qwen2.5 attention blocks without forwarding the override.
        with honor_qwen_sdpa(attention_module,requested,vllm_version=vllm_version,inherited_override=True):
            yield
    finally:module.check_upstream_fa_availability=original
