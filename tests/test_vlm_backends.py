import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from jsonschema.exceptions import ValidationError
from rosa_repro.adapter import ProtocolError
from rosa_repro.component_runtime import ComponentCore,ComponentResult,validate_output
from rosa_repro.dummy import observation
from rosa_repro.vlm_contract import (default_config,runtime_fingerprint,validate_config,prepare_request,parse_generation,verify_snapshot)
from rosa_repro.vlm_backends import (TransformersVlmBackend,VllmVlmBackend,parsed_result,vllm_engine_options,validate_worker)


def payload(value=1):
    o=observation(value);o['video.image']=np.zeros((1,2,28,28,3),np.uint8);o['video.image'][0,-1]=value
    o['video.wrist_image']=np.full((1,1,28,28,3),2*value,np.uint8)
    return dict(observation=o,context={'execution_history':'2 completed chunks'},plan='move left')


def worker(c):return dict(worker_id='w',component_id=c['role'],role=c['role'],model_revision=c['revision'],
    runtime_fingerprint=runtime_fingerprint(c),batch_size=1,batch_wait_ms=0,resources={'num_gpus':1},members=['r'])


class Processor:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=False,add_generation_prompt=True)
        return messages[0]['content']+' '+messages[1]['content'][-1]['text']


@pytest.mark.parametrize('role,text,expected',[
 ('s2','{"plan":"pick cup"}',{'plan':'pick cup'}),
 ('safety','{"decision":"safe","reason":"clear"}',{'stop':False,'reason':'clear'}),
 ('safety','{"decision":"unsafe","reason":"person"}',{'stop':True,'reason':'person'}),
 ('safety','{"decision":"unknown","reason":"occluded"}',{'stop':True,'reason':'occluded'}),
 ('monitor','{"status":"done","reason":"placed"}',{'text':'placed','task_status':'done'}),
 ('monitor','{"status":"unknown","reason":"no view"}',{'text':'no view','task_status':'unknown'})])
def test_strict_role_outputs(role,text,expected):
    assert parse_generation(role,text,'stop')==expected;validate_output(role,expected)


@pytest.mark.parametrize('text,finish',[
 ('{"decision": "safe", "reason": "clear"}','length'),
 ('{"decision": "safe", "reason": "clear"}','abort'),
 ('not unsafe','stop'),('```json\n{"decision":"safe","reason":"ok"}\n```','stop'),
 ('{"decision":"safe","reason":"a","decision":"unsafe"}','stop'),
 ('{"decision":true,"reason":"ok"}','stop'),('{"decision":"safe","reason":""}','stop'),
 ('{"decision":"SAFE","reason":"ok"}','stop'),('{"decision":"safe","reason":"ok","confidence":1}','stop'),
 ('{"decision":"safe","reason":NaN}','stop'),('[]','stop'),('', 'stop')])
def test_invalid_safety_generation_is_never_safe(text,finish):
    c=default_config('safety');r=parsed_result(c,dict(text=text,finish_reason=finish,generated_tokens=7),{})
    assert r.output is None and r.error_type=='InvalidModelGeneration' and r.evidence['raw_generation']==text


@pytest.mark.parametrize('change',['revision','engine','pixels','role_size','concurrency','fast','output_limit'])
def test_configs_fail_closed_and_fingerprint_changes(change):
    c=default_config('s2');a=runtime_fingerprint(c);c['max_new_tokens']+=1;assert runtime_fingerprint(c)!=a
    if change=='revision':c['revision']='0'*40
    elif change=='engine':c['attention_backend']='vllm_auto'
    elif change=='pixels':c['min_pixels']=c['max_pixels']+1
    elif change=='role_size':c['role']='safety'
    elif change=='concurrency':c['max_concurrent_requests']=2
    elif change=='fast':c['processor_use_fast']=True
    else:c['max_new_tokens']=c['max_model_len']
    with pytest.raises((ValueError,ValidationError)):validate_config(c)


def test_images_latest_frame_order_rgb_and_input_immutable():
    c=default_config('monitor');p=payload(3);before=p['observation']['video.image'].copy()
    messages,images,trace=prepare_request(p,c)
    assert np.array(images[0]).shape==(28,28,3) and (np.array(images[0])==3).all()
    assert (np.array(images[1])==6).all() and trace['images'][0]['selected_frame']==1
    assert np.array_equal(p['observation']['video.image'],before)
    assert trace['context_fields']==['execution_history'] and len(trace['prompt_sha256'])==64
    text=json.loads(messages[1]['content'][-1]['text']);assert text['current_plan']=='move left'


@pytest.mark.parametrize('change',['missing_camera','float','batch','layout','task','context','long'])
def test_bad_observations_refused_without_hidden_conversion(change):
    c=default_config('s2');p=payload()
    if change=='missing_camera':p['observation'].pop('video.wrist_image')
    elif change=='float':p['observation']['video.image']=p['observation']['video.image'].astype(np.float32)
    elif change=='batch':p['observation']['video.image']=np.zeros((2,1,28,28,3),np.uint8)
    elif change=='layout':p['observation']['video.image']=np.zeros((1,1,3,28,28),np.uint8)
    elif change=='task':p['observation']['task']=['a','b']
    elif change=='context':p['context']={'untracked':'data'}
    else:p['observation']['task']=['x'*10000]
    with pytest.raises(ProtocolError):prepare_request(p,c)


def test_checkpoint_hashes_not_folder_names_are_proof(tmp_path,monkeypatch):
    import hashlib
    import rosa_repro.vlm_contract as module
    data=b'processor';weight=b'weight'
    catalog={'test':dict(revision='1'*40,files={n:dict(sha256=hashlib.sha256(d).hexdigest(),size=len(d),kind=k)
        for n,d,k in [('config.json',data,'processor_or_config'),('weights.safetensors',weight,'weight')]})}
    monkeypatch.setattr(module,'pins',lambda:catalog)
    (tmp_path/'config.json').write_bytes(data)
    assert not verify_snapshot(tmp_path,'test',weights=False)['full_weight_hashes_verified']
    with pytest.raises(ValueError,match='Missing'):verify_snapshot(tmp_path,'test')
    (tmp_path/'weights.safetensors').write_bytes(weight)
    assert verify_snapshot(tmp_path,'test')['full_weight_hashes_verified']
    (tmp_path/'weights.safetensors').write_bytes(b'WEIGHT')
    with pytest.raises(ValueError,match='hash'):verify_snapshot(tmp_path,'test')


class Engine:
    def __init__(self,text='{"status":"ongoing","reason":"still moving"}',finish='stop',delay=.01,wrong_id=False):
        self.text=text;self.finish=finish;self.delay=delay;self.wrong_id=wrong_id;self.active=0;self.peak=0;self.aborted=[];self.ids=[]
    async def generate(self,prompt,params,request_id):
        assert len(prompt['multi_modal_data']['image'])==2 and params.temperature==0
        self.ids.append(request_id);self.active+=1;self.peak=max(self.peak,self.active)
        try:
            await asyncio.sleep(self.delay)
            yield SimpleNamespace(request_id='wrong' if self.wrong_id else request_id,finished=True,
                outputs=[SimpleNamespace(text=self.text,finish_reason=self.finish,token_ids=[1,2,3])])
        finally:self.active-=1
    async def abort(self,request_id):self.aborted.append(request_id)


def async_backend(**kwargs):
    c=default_config('monitor','vllm');c['max_concurrent_requests']=2
    if 'timeout' in kwargs:c['generation_timeout_s']=kwargs.pop('timeout')
    e=Engine(**kwargs)
    b=VllmVlmBackend(worker(c),c,Processor(),e,lambda **kw:SimpleNamespace(**kw),{'checkpoint':{'full_weight_hashes_verified':False}})
    return b,e


def test_async_engine_concurrent_requests_have_unique_ids_and_bounded_admission():
    b,e=async_backend()
    async def go():return await asyncio.gather(*(b.predict_batch([payload(i)]) for i in range(4)))
    rs=asyncio.run(go())
    assert e.peak==2 and len(set(e.ids))==4 and not e.aborted
    assert all(x[0].output['task_status']=='ongoing' for x in rs)


def test_async_engine_timeout_aborts_and_never_returns_partial_json():
    b,e=async_backend(delay=.1,timeout=.005)
    result=asyncio.run(b.predict_batch([payload()]))[0]
    assert result.output is None and result.error_type=='ModelGenerationTimeout'
    assert len(e.aborted)==1 and e.active==0


def test_async_identity_mismatch_and_truncated_output():
    b,e=async_backend(wrong_id=True)
    result=asyncio.run(b.predict_batch([payload()]))[0]
    assert result.output is None and result.evidence['error_type']=='ProtocolError'
    assert e.aborted
    b,e=async_backend(finish='length')
    r=asyncio.run(b.predict_batch([payload()]))[0];assert r.output is None and r.evidence['finish_reason']=='length'


def test_reference_and_async_use_same_parser_and_worker_identity():
    c=default_config('s2');w=worker(c)
    generator=SimpleNamespace(generate=lambda p,i:dict(text='{"plan":"pick cup"}',finish_reason='stop',generated_tokens=5))
    b=TransformersVlmBackend(w,c,Processor(),generator,{'checkpoint':{'full_weight_hashes_verified':False}})
    r=b.predict_batch([payload()])[0];assert r.output=={'plan':'pick cup'} and r.evidence['parse_status']=='ok'
    w['runtime_fingerprint']='wrong'
    with pytest.raises(ValueError,match='fingerprint'):validate_worker(w,c)


def test_engine_options_match_resource_and_cache_contract():
    c=default_config('safety','vllm');o=vllm_engine_options(c,'/local/model')
    assert o['tensor_parallel_size']==o['pipeline_parallel_size']==1
    assert o['mm_processor_kwargs']['use_fast'] is False and o['mm_processor_cache_gb']==0
    assert o['enable_prefix_caching'] is False and o['max_num_seqs']==16


@pytest.mark.parametrize('decision,stop',[('safe',False),('unsafe',True),('unknown',True)])
def test_label_only_safety_keeps_unknown_fail_closed(decision,stop):
    from rosa_repro.vlm_contract import generation_schema
    c=default_config('safety','vllm');c['structured_output']='safety_label_v1'
    validate_config(c)
    assert generation_schema('safety','safety_label_v1')['required']==['decision']
    r=parsed_result(c,dict(text=json.dumps({'decision':decision}),finish_reason='stop',generated_tokens=6),{})
    assert r.output['stop'] is stop and 'no explanation generated' in r.output['reason']
    assert parsed_result(c,dict(text='{"decision":"safe"}',finish_reason='length',generated_tokens=6),{}).output is None
    assert parsed_result(c,dict(text='{"decision":"safe","reason":"clear"}',finish_reason='stop',generated_tokens=10),{}).output is None


def test_host_stream_timing_records_first_token_separately():
    b,e=async_backend()
    async def streaming(prompt,params,request_id):
        await asyncio.sleep(.004)
        yield SimpleNamespace(request_id=request_id,finished=False,outputs=[SimpleNamespace(token_ids=[1])])
        await asyncio.sleep(.004)
        yield SimpleNamespace(request_id=request_id,finished=True,prompt_token_ids=[1]*20,
            outputs=[SimpleNamespace(text=e.text,finish_reason='stop',token_ids=[1,2,3])])
    e.generate=streaming
    r=asyncio.run(b.predict_batch([payload()]))[0]
    t=r.evidence['latency_breakdown']
    assert t['first_token_ms']>0 and t['after_first_token_ms']>0
    assert t['engine_stream_ms']>=t['first_token_ms'] and r.evidence['prompt_tokens']==20
    assert not t['isolated_vision_or_prefill_cuda_time_available']


@pytest.mark.parametrize('status',['ongoing','done','failed','unknown'])
def test_monitor_label_output_is_explicit_and_preserves_status(status):
    from rosa_repro.vlm_contract import generation_schema
    c=default_config('monitor','vllm');c['structured_output']='monitor_label_v1'
    validate_config(c)
    assert generation_schema('monitor','monitor_label_v1')['required']==['status']
    assert vllm_engine_options(c,'/local')['structured_outputs_config']['disable_any_whitespace']
    r=parsed_result(c,dict(text=json.dumps({'status':status}),finish_reason='stop',generated_tokens=7),{})
    validate_output('monitor',r.output)
    assert r.output['task_status']==status and 'no explanation generated' in r.output['text']
    for text,finish in [('{}','stop'),('{"status":"done","reason":"ok"}','stop'),('{"status":"done"}','length'),('{"status":"success"}','stop')]:
        assert parsed_result(c,dict(text=text,finish_reason=finish,generated_tokens=7),{}).output is None


def test_short_s2_requires_complete_bounded_plan_and_correct_role():
    from rosa_repro.vlm_contract import generation_schema
    c=default_config('s2','vllm');c['structured_output']='s2_short_v1';validate_config(c)
    assert generation_schema('s2','s2_short_v1')['properties']['plan']['maxLength']==64
    for text,finish,accepted in [('{"plan":"pick cup"}','stop',True),('{"plan":"pick cup"}','length',False),
        (json.dumps({'plan':'x'*65}),'stop',False),('{"plan":""}','stop',False)]:
        result=parsed_result(c,dict(text=text,finish_reason=finish,generated_tokens=9),{})
        assert (result.output is not None)==accepted
    for role,format in [('safety','monitor_label_v1'),('s2','monitor_label_v1'),('monitor','s2_short_v1')]:
        c=default_config(role,'vllm');c['structured_output']=format
        with pytest.raises(ValueError):validate_config(c)
        with pytest.raises(ValueError):generation_schema(role,format)


def test_structured_decoding_is_explicit_without_unconstrained_fallback():
    from rosa_repro.vlm_contract import generation_schema
    from jsonschema import validate
    c=default_config('monitor','vllm');old=runtime_fingerprint(c)
    c['structured_output']='role_json_v1'
    assert runtime_fingerprint(c)!=old
    assert vllm_engine_options(c,'/local/model')['structured_outputs_config']==dict(backend='xgrammar',disable_fallback=True)
    e=Engine();captured=[]
    def sampling(**kw):captured.append(kw);return SimpleNamespace(**kw)
    with pytest.raises(ValueError,match='fallback'):
        VllmVlmBackend(worker(c),c,Processor(),e,sampling,{})
    b=VllmVlmBackend(worker(c),c,Processor(),e,sampling,{},lambda **kw:SimpleNamespace(**kw))
    r=asyncio.run(b.predict_batch([payload()]))[0]
    assert r.output['task_status']=='ongoing'
    spec=captured[0]['structured_outputs']
    assert spec.disable_fallback and spec.json==generation_schema('monitor')
    assert r.evidence['structured_generation']['semantic_accuracy_verified'] is False
    for value in [{'status':'invented','reason':'x'},{'status':'done','reason':'x','extra':1},{'status':'done','reason':'x'*193}]:
        with pytest.raises(ValidationError):validate(value,spec.json)
    e.finish='length'
    assert asyncio.run(b.predict_batch([payload()]))[0].error_type=='InvalidModelGeneration'
    hf=default_config('safety');hf['structured_output']='role_json_v1'
    with pytest.raises(ValueError,match='vLLM'):validate_config(hf)
    original=runtime_fingerprint(c);c['structured_output']='role_json_compact_v1'
    assert runtime_fingerprint(c)!=original
    assert vllm_engine_options(c,'/local/model')['structured_outputs_config']['disable_any_whitespace'] is True
    b=VllmVlmBackend(worker(c),c,Processor(),e,sampling,{},lambda **kw:SimpleNamespace(**kw))
    asyncio.run(b.predict_batch([payload()]))
    assert captured[-1]['structured_outputs'].disable_any_whitespace is True


def test_component_core_keeps_invalid_raw_generation_in_response(monkeypatch):
    import rosa_repro.component_runtime as module
    c=default_config('safety');w=worker(c);w['backend']={'factory':'test_module:factory','kwargs':{}}
    class Backend:
        def describe(self):return dict(execution_kind='external_model',stateless=True,**{k:w[k] for k in ('component_id','role','model_revision','runtime_fingerprint')})
        def predict_batch(self,payloads):return [parsed_result(c,dict(text='safe',finish_reason='stop',generated_tokens=1),{})]
    monkeypatch.setattr(module.importlib,'import_module',lambda name:SimpleNamespace(factory=lambda **kw:Backend()))
    core=ComponentCore(w,'plan');ack=core.reset('run','r','e')
    meta=dict(plan_id='plan',worker_id='w',run_id='run',robot_id='r',episode_id='e',request_id='q',attempt_id='a',
              component_id='safety',observation_seq=0,generation=ack['generation'])
    t=core.accept(meta,payload());r=asyncio.run(core.run_batch([t]))[0]
    assert r['status']=='error' and r['output'] is None and r['backend_evidence']['raw_generation']=='safe'
