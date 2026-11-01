import asyncio,gzip,json,sys
from pathlib import Path
import pytest
from rosa_repro.runtime_attestation import (static_environment,collect_environment,model_state,torch_state,
    summarize_trace,capture_torch,capture_vllm)


def test_environment_allowlist_excludes_credentials_and_records_binary(monkeypatch):
    monkeypatch.setenv('HF_TOKEN','never-record-this');monkeypatch.setenv('RAY_AUTH_TOKEN','never-record-this')
    monkeypatch.setenv('CUDA_MODULE_LOADING','LAZY')
    x=static_environment();assert 'never-record-this' not in json.dumps(x)
    assert x['selected_environment']['CUDA_MODULE_LOADING']=='LAZY' and len(x['python_binary_sha256'])==64
    assert x==static_environment()


def test_missing_tools_and_torch_unrequested_are_explicit(monkeypatch):
    monkeypatch.setattr('rosa_repro.runtime_attestation.shutil.which',lambda name:None)
    x=collect_environment();assert all(v['status']=='unavailable' for v in x['tools'].values())
    assert x['torch']['status']=='not_requested' and not x['kernel_trace_collected']


def test_trace_CPU_api_names_are_not_GPU_kernel_execution(tmp_path):
    p=tmp_path/'cpu.json';p.write_text(json.dumps({'traceEvents':[dict(ph='X',cat='cpu_op',name='cudaLaunchKernel',dur=10),
        dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',dur=20)]}))
    r=summarize_trace(p);assert not r['gpu_kernel_events_present']
    p.write_text(json.dumps({'traceEvents':[dict(ph='X',cat='kernel',name='test_only_attention_kernel',dur=4,pid=1)]}))
    r=summarize_trace(p);assert r['gpu_kernel_events']==1 and not r['optimization_equivalence_verified']


@pytest.mark.parametrize('bad',['nonfinite','negative','shape','gzip_limit'])
def test_trace_rejects_corrupt_or_unbounded_data(tmp_path,bad):
    p=tmp_path/'x.json';data={'traceEvents':[dict(ph='X',cat='kernel',name='fixture',dur=-1 if bad=='negative' else 1)]}
    if bad=='nonfinite':data['traceEvents'][0]['dur']=float('nan')
    if bad=='shape':data={'traceEvents':{}}
    if bad=='gzip_limit':p=tmp_path/'x.json.gz';p.write_bytes(gzip.compress(b' '*5000))
    else:p.write_text(json.dumps(data))
    with pytest.raises(ValueError):summarize_trace(p,max_bytes=1024)


def test_real_CPU_torch_profiler_and_model_attributes(tmp_path):
    torch=pytest.importorskip('torch');model=torch.nn.Linear(4,4).eval();x=torch.ones(2,4)
    state=model_state(model);assert state['parameter_elements_by_dtype']=={'torch.float32':20}
    assert state['complete_attention_kernel_identity'] is None and state['cuda_graph_execution'] is None
    value,trace=capture_torch(lambda:model(x),tmp_path/'trace.json')
    assert tuple(value.shape)==(2,4) and trace['cpu_operator_names'] and not trace['gpu_kernel_events_present']
    with pytest.raises(FileExistsError):capture_torch(lambda:model(x),tmp_path/'trace.json')
    if not torch.cuda.is_available():
        with pytest.raises(ValueError,match='no CPU fallback'):capture_torch(lambda:model(x),tmp_path/'cuda.json',cuda=True)
    assert torch_state(torch)['enabled_backends_are_not_observed_kernels']


@pytest.mark.parametrize('failure',[False,True])
def test_native_vllm_profiler_always_stopped_and_frontend_only_is_insufficient(tmp_path,failure):
    class Engine:
        def __init__(self):self.started=False;self.stopped=False
        async def start_profile(self):self.started=True
        async def stop_profile(self):
            self.stopped=True;(tmp_path/'frontend.json').write_text(json.dumps({'traceEvents':[dict(ph='X',cat='cpu_op',name='frontend',dur=1)]}))
    e=Engine()
    async def call():
        if failure:raise RuntimeError('injected generation failure')
        return 'test fixture'
    if failure:
        with pytest.raises(RuntimeError):asyncio.run(capture_vllm(e,call,tmp_path))
    else:
        _,traces=asyncio.run(capture_vllm(e,call,tmp_path));assert not any(x['gpu_kernel_events_present'] for x in traces.values())
    assert e.started and e.stopped


@pytest.mark.parametrize('stop',['timeout','interrupt'])
def test_supervisor_stops_own_process_group_and_preserves_failed_evidence(tmp_path,monkeypatch,stop):
    import importlib.util,subprocess,signal
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'scripts/runtime_evidence.py'
    spec=importlib.util.spec_from_file_location('kernel_cli_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Process:
        pid=12345
        def __init__(self,*a,**kw):self.calls=0;assert kw['start_new_session']
        def wait(self,timeout):
            self.calls+=1
            if self.calls==1:
                if stop=='interrupt':raise KeyboardInterrupt()
                raise subprocess.TimeoutExpired('synthetic',timeout)
            return -15
    signals=[];monkeypatch.setattr(m.subprocess,'Popen',Process);monkeypatch.setattr(m.os,'killpg',lambda pid,s:signals.append((pid,s)))
    a=SimpleNamespace(output=tmp_path/'out',cpu_self_test=True,timeout=10,flush_timeout=1)
    assert m.supervise(a)==2
    result=json.loads((a.output/'result.json').read_text());assert result['status']==('timeout' if stop=='timeout' else 'interrupted')
    assert not result['gpu_kernel_events_present'] and signals==[(12345,signal.SIGTERM),(12345,signal.SIGKILL)]
