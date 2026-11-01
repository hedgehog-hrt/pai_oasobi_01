"""Real Qwen-VL factories: Transformers reference and vLLM V1 async serving.
All model loads are local-only; use a prepared, hash-verified pinned snapshot.
"""
import asyncio
from copy import deepcopy
from importlib.metadata import version
import inspect
from pathlib import Path
import time
import uuid
import os

from .adapter import ProtocolError
from .component_runtime import ComponentResult
from .execution_plan import digest
from .vlm_contract import validate_config,runtime_fingerprint,verify_snapshot,prepare_request,parse_generation,generation_schema,COMPACT_FORMATS,ROLE_FORMATS

EXPECTED_VERSIONS={'transformers':'4.57.6','torch':'2.9.0','Pillow':'12.0.0'}


def check_versions(engine,runtime_variant=None):
    expected=dict(EXPECTED_VERSIONS)
    if runtime_variant is not None:
        from .shared_environment import VARIANT
        if runtime_variant!=VARIANT:raise ValueError('Unknown VLM runtime variant')
        expected['transformers']='4.57.3'
    if engine=='vllm':expected['vllm']='0.11.2'
    actual={k:version(k) for k in expected}
    if any(actual[k].split('+')[0]!=v for k,v in expected.items()):raise ValueError('VLM dependency versions differ: '+str(actual))
    return actual


def validate_worker(worker,config,runtime_variant=None):
    validate_config(config)
    if worker['role']!=config['role'] or worker['model_revision']!=config['revision'] or worker['runtime_fingerprint']!=runtime_fingerprint(config,runtime_variant):
        raise ValueError('Worker identity/profile fingerprint differs from VLM configuration')
    if worker['batch_size']!=1 or worker['batch_wait_ms']!=0:raise ValueError('VLM uses batch-one admission; vLLM batches internally')


def rendered_request(payload,config,processor):
    messages,images,trace=prepare_request(payload,config)
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    if not isinstance(prompt,str) or not prompt:raise ProtocolError('Processor returned an empty prompt')
    trace['rendered_prompt_sha256']=digest(prompt)
    return prompt,images,trace


def parsed_result(config,raw,trace):
    evidence=dict(trace,raw_generation=raw['text'],finish_reason=raw['finish_reason'],
                  generated_tokens=raw['generated_tokens'],generation_parameters=raw.get('parameters',{}),
                  parser='strict-role-json-v1')
    try:
        output=parse_generation(config['role'],raw['text'],raw['finish_reason'],config.get('structured_output','none'))
        if config.get('structured_output') in ROLE_FORMATS:evidence['parser']='strict-'+config['structured_output'].replace('_','-')
        evidence['parse_status']='ok'
        return ComponentResult(output,evidence)
    except ProtocolError as exc:
        evidence.update(parse_status='invalid',parse_error=str(exc))
        return ComponentResult(None,evidence,'InvalidModelGeneration')


class HfGenerator:
    """Generation adapter also exercised by a tiny random CPU Qwen architecture."""
    def __init__(self,model,processor,config,device):
        self.model,self.processor,self.config,self.device=model,processor,config,device
    def generate(self,prompt,images):
        import torch
        inputs=self.processor(text=[prompt],images=images,padding=True,return_tensors='pt')
        width=inputs['input_ids'].shape[1]
        if width+self.config['max_new_tokens']>self.config['max_model_len']:
            raise ProtocolError('Multimodal prompt exceeds context budget; no truncation')
        inputs=inputs.to(self.device)
        generation=deepcopy(self.model.generation_config)
        generation.do_sample=False;generation.num_beams=1;generation.num_return_sequences=1
        generation.temperature=None;generation.top_p=None;generation.top_k=None
        generation.repetition_penalty=1.;generation.no_repeat_ngram_size=0
        generation.max_new_tokens=self.config['max_new_tokens'];generation.max_time=self.config['generation_timeout_s']
        generation.return_dict_in_generate=False;generation.output_scores=False;generation.use_cache=True
        generation.pad_token_id=self.processor.tokenizer.pad_token_id
        with torch.inference_mode():ids=self.model.generate(**inputs,generation_config=generation)
        if ids.shape[0]!=1:raise ProtocolError('Unexpected generated sequence count')
        new=ids[0,width:];tokens=new.tolist()
        eos=generation.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
        finish='stop' if tokens and tokens[-1] in eos else 'length' if len(tokens)>=generation.max_new_tokens else 'aborted'
        text=self.processor.batch_decode([tokens],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        return dict(text=text,finish_reason=finish,generated_tokens=len(tokens),
                    parameters=dict(do_sample=False,num_beams=1,max_new_tokens=generation.max_new_tokens,
                                    use_cache=True,input_tokens=width,max_time_s=generation.max_time))


class TransformersVlmBackend:
    def __init__(self,worker,config,processor,generator,attestation):
        validate_worker(worker,config)
        self.worker,self.config,self.processor,self.generator=worker,deepcopy(config),processor,generator
        self.attestation=attestation
    def describe(self):
        return dict(execution_kind='external_model',stateless=True,component_id=self.worker['component_id'],role=self.worker['role'],
            model_revision=self.config['revision'],runtime_fingerprint=runtime_fingerprint(self.config),
            max_concurrent_requests=1,serving_mode='serialized_transformers_reference',
            model_artifacts_verified=self.attestation['checkpoint']['full_weight_hashes_verified'],
            config=self.config,attestation=self.attestation)
    def predict_batch(self,payloads):
        if len(payloads)!=1:raise ProtocolError('Reference VLM accepts exactly one request per call')
        prompt,images,trace=rendered_request(payloads[0],self.config,self.processor)
        return [parsed_result(self.config,self.generator.generate(prompt,images),trace)]


def vllm_engine_options(config,model_path):
    validate_config(config)
    options=dict(model=str(model_path),tokenizer=str(model_path),dtype=config['dtype'],trust_remote_code=False,
        tensor_parallel_size=1,pipeline_parallel_size=1,distributed_executor_backend='mp',
        max_model_len=config['max_model_len'],max_num_seqs=config['max_concurrent_requests'],
        gpu_memory_utilization=config['gpu_memory_utilization'],seed=config['seed'],
        enable_prefix_caching=False,mm_processor_cache_gb=0,enforce_eager=config['enforce_eager'],
        limit_mm_per_prompt={'image':len(config['image_keys']),'video':0},
        mm_processor_kwargs=dict(min_pixels=config['min_pixels'],max_pixels=config['max_pixels'],use_fast=config['processor_use_fast']),
        enable_chunked_prefill=False,disable_log_stats=True,enable_log_requests=False)
    if config.get('vision_attention_backend','auto')=='TORCH_SDPA':
        options.update(mm_encoder_attn_backend='TORCH_SDPA',worker_cls='rosa_repro.vllm_worker.QwenSdpaWorker')
    if 'mps_client_priority' in config:
        options['additional_config']={'rosa_mps_priority':dict(priority=config['mps_client_priority'],role=config['role'])}
    if config.get('structured_output','none')!='none':
        options['structured_outputs_config']=dict(backend='xgrammar',disable_fallback=True)
        if config['structured_output'] in COMPACT_FORMATS:options['structured_outputs_config']['disable_any_whitespace']=True
    return options


class VllmVlmBackend:
    def __init__(self,worker,config,processor,engine,sampling_factory,attestation,structured_factory=None):
        validate_worker(worker,config,attestation.get('runtime_variant'))
        self.worker,self.config,self.processor=worker,deepcopy(config),processor
        self.engine,self.sampling_factory,self.attestation=engine,sampling_factory,attestation
        self.structured_factory=structured_factory
        if config.get('structured_output','none')!='none' and structured_factory is None:
            raise ValueError('Structured output factory is required; unconstrained fallback is disabled')
        self.semaphore=asyncio.Semaphore(config['max_concurrent_requests'])
    def describe(self):
        return dict(execution_kind='external_model',stateless=True,component_id=self.worker['component_id'],role=self.worker['role'],
            model_revision=self.config['revision'],runtime_fingerprint=runtime_fingerprint(self.config,self.attestation.get('runtime_variant')),
            max_concurrent_requests=self.config['max_concurrent_requests'],serving_mode='vllm_continuous',
            model_artifacts_verified=self.attestation['checkpoint']['full_weight_hashes_verified'],
            config=self.config,attestation=self.attestation,
            internal_token_batch_sizes_observed=False)
    async def predict_batch(self,payloads):
        if len(payloads)!=1:raise ProtocolError('vLLM worker admits individual requests for engine scheduling')
        begin=time.monotonic_ns()
        prompt,images,trace=rendered_request(payloads[0],self.config,self.processor)
        prepared=time.monotonic_ns()
        request_id='vlm-'+uuid.uuid4().hex
        structured={}
        if self.config.get('structured_output','none')!='none':
            schema=generation_schema(self.config['role'],self.config['structured_output'])
            kw=dict(json=schema,disable_fallback=True)
            if self.config['structured_output'] in COMPACT_FORMATS:kw['disable_any_whitespace']=True
            structured['structured_outputs']=self.structured_factory(**kw)
            trace['structured_generation']=dict(format=self.config['structured_output'],backend='xgrammar',schema=schema,schema_sha256=digest(schema),semantic_accuracy_verified=False)
        params=self.sampling_factory(n=1,temperature=0.,top_p=1.,top_k=-1,seed=self.config['seed'],
                                     max_tokens=self.config['max_new_tokens'],ignore_eos=False,**structured)
        final=None;first_token_ns=None;submitted=None;waiting=time.monotonic_ns()
        def timing():
            finished=time.monotonic_ns()
            return dict(protocol='vlm_host_stream_timing_v1',clock='backend-process-monotonic',
                request_preparation_ms=(prepared-begin)/1e6,
                admission_wait_ms=(submitted-waiting)/1e6 if submitted else None,
                first_token_ms=(first_token_ns-submitted)/1e6 if first_token_ns and submitted else None,
                after_first_token_ms=(finished-first_token_ns)/1e6 if first_token_ns else None,
                engine_stream_ms=(finished-submitted)/1e6 if submitted else None,
                total_ms=(finished-begin)/1e6,
                first_token_scope='first nonempty token output delivered by async engine; includes engine queue, input processing and prefill',
                isolated_vision_or_prefill_cuda_time_available=False)
        try:
            async with asyncio.timeout(self.config['generation_timeout_s']):
                async with self.semaphore:
                    submitted=time.monotonic_ns()
                    async for response in self.engine.generate(dict(prompt=prompt,multi_modal_data={'image':images}),params,request_id):
                        final=response
                        if first_token_ns is None and any(getattr(o,'token_ids',[]) for o in response.outputs):first_token_ns=time.monotonic_ns()
            if final is None or not final.finished or len(final.outputs)!=1:raise ProtocolError('Incomplete vLLM stream')
            if final.request_id!=request_id:raise ProtocolError('vLLM request identity mismatch')
            completion=final.outputs[0]
            trace['latency_breakdown']=timing()
            trace['prompt_tokens']=len(final.prompt_token_ids) if getattr(final,'prompt_token_ids',None) is not None else None
            raw=dict(text=completion.text,finish_reason=completion.finish_reason,generated_tokens=len(completion.token_ids),
                     parameters=dict(temperature=0.,seed=self.config['seed'],max_new_tokens=self.config['max_new_tokens'],
                                     engine_request_id=request_id,prefix_caching=False,multimodal_processor_cache_gb=0))
            return [parsed_result(self.config,raw,trace)]
        except asyncio.CancelledError:
            await asyncio.shield(self.engine.abort(request_id))
            raise
        except Exception as exc:
            abort_error=None
            try:await asyncio.shield(self.engine.abort(request_id))
            except Exception as cleanup:abort_error=type(cleanup).__name__
            partial=final.outputs[0] if final is not None and len(final.outputs)==1 else None
            trace.update(raw_generation=partial.text if partial else '',generated_tokens=len(partial.token_ids) if partial else 0,
                finish_reason='timeout' if isinstance(exc,TimeoutError) else 'error',parse_status='not_accepted',
                engine_request_id=request_id,error_type=type(exc).__name__,abort_error_type=abort_error)
            trace['latency_breakdown']=timing()
            return [ComponentResult(None,trace,'ModelGenerationTimeout' if isinstance(exc,TimeoutError) else 'ModelGenerationError')]


def load_common(worker,config,model_path,expected_engine,runtime_variant=None):
    validate_worker(worker,config,runtime_variant)
    if config['engine']!=expected_engine:raise ValueError('Factory/config engine mismatch')
    path=Path(model_path).resolve()
    checkpoint=verify_snapshot(path,config['model_id'],weights=True)
    versions=check_versions(expected_engine,runtime_variant)
    import torch
    from .runtime_attestation import torch_state
    if worker['resources']['num_gpus']!=1 or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Real VLM factory requires exactly one Ray-visible CUDA device')
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
        min_pixels=config['min_pixels'],max_pixels=config['max_pixels'],use_fast=config['processor_use_fast'])
    processor.tokenizer.padding_side='left'
    attestation=dict(checkpoint=checkpoint,versions=versions,torch_cuda=torch.version.cuda,
        observed_torch_state=torch_state(torch),
        gpu_name=torch.cuda.get_device_name(0),device='cuda:0',configured_runtime_sha256=runtime_fingerprint(config,runtime_variant),
        kernel_optimization_verified=False,paper_configuration_equivalence_verified=False)
    if runtime_variant is not None:attestation['runtime_variant']=runtime_variant
    return path,processor,attestation


def transformers_vlm_backend(worker,config,model_path):
    path,processor,attestation=load_common(worker,config,model_path,'transformers')
    import torch
    import transformers
    name='Qwen3VLForConditionalGeneration' if config['model_id'].startswith('Qwen/Qwen3-VL-') else 'Qwen2_5_VLForConditionalGeneration'
    model=getattr(transformers,name).from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
        torch_dtype=getattr(torch,config['dtype']),attn_implementation=config['attention_backend'])
    model=model.to('cuda:0').eval()
    from collections import Counter
    attestation['parameter_dtypes']=dict(Counter(str(p.dtype) for p in model.parameters()))
    from .runtime_attestation import model_state
    attestation['observed_model_state']=model_state(model)
    return TransformersVlmBackend(worker,config,processor,HfGenerator(model,processor,config,'cuda:0'),attestation)


def vllm_vlm_backend(worker,config,model_path,runtime_variant=None):
    path,processor,attestation=load_common(worker,config,model_path,'vllm',runtime_variant)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ['HF_HUB_OFFLINE']='1'
    from vllm import AsyncEngineArgs,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.engine.async_llm import AsyncLLM
    options=vllm_engine_options(config,path);attestation['engine_options']=options
    if options.get('worker_cls')=='rosa_repro.vllm_worker.QwenSdpaWorker':
        family='qwen3' if config['model_id'].startswith('Qwen/Qwen3-VL-') else 'qwen'
        attestation['compatibility_policy']='vllm-0.11.2-'+family+'-explicit-vision-sdpa-v1'
    attestation['worker_multiprocess_method']='spawn'
    if 'mps_client_priority' in config:
        from .mps_priority import engine_launch
        with engine_launch(options['additional_config']['rosa_mps_priority']) as evidence:
            engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
            attestation['mps_launch']=evidence
    else:engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
    return VllmVlmBackend(worker,config,processor,engine,SamplingParams,attestation,StructuredOutputsParams)
