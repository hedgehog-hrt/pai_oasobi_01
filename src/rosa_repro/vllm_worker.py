"""Opt-in vLLM GPU worker; imported only in a configured real-model process."""
from importlib.metadata import version
from vllm.v1.worker.gpu_worker import Worker
from .vision_compat import honor_qwen_sdpa,honor_qwen3_sdpa


class QwenSdpaWorker(Worker):
    def init_device(self):
        settings=self.vllm_config.additional_config.get('rosa_mps_priority')
        if settings is not None:
            import json
            from .mps_priority import attest
            print('ROSA_MPS_CLIENT '+json.dumps(attest(settings),sort_keys=True),flush=True)
        return super().init_device()

    def load_model(self):
        config=self.vllm_config.model_config
        if config.hf_config.architectures==['Qwen2_5_VLForConditionalGeneration']:
            from vllm.model_executor.models import qwen2_5_vl as module
            compat=honor_qwen_sdpa
            kwargs={}
        elif config.hf_config.architectures==['Qwen3VLForConditionalGeneration']:
            from vllm.model_executor.models import qwen3_vl as module
            from vllm.model_executor.models import qwen2_5_vl
            compat=honor_qwen3_sdpa
            kwargs={'attention_module':qwen2_5_vl}
        else:raise ValueError('Vision compatibility worker requires a supported Qwen VL architecture')
        requested=config.multimodal_config.mm_encoder_attn_backend
        with compat(module,requested,vllm_version=version('vllm'),**kwargs):
            result=super().load_model()
        visual=self.model_runner.model.visual
        if visual.attn_backend!=requested or any(b.attn.attn_backend!=requested for b in visual.blocks):
            raise ValueError('Loaded vision attention differs from declared TORCH_SDPA')
        return result
