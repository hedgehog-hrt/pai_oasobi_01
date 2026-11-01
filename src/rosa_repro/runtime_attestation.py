"""Read-only environment/model observations and bounded Chrome trace inspection."""
from collections import Counter
from importlib import metadata
from pathlib import Path
import csv,gzip,hashlib,json,math,os,platform,shutil,subprocess,sys
from .execution_plan import digest,runtime_source_manifest

# Deliberately omit arbitrary environment, credentials, pip URLs and process argv.
ENV_KEYS=('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','CUBLAS_WORKSPACE_CONFIG',
 'NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF',
 'VLLM_ATTENTION_BACKEND','VLLM_USE_V1','VLLM_WORKER_MULTIPROC_METHOD','VLLM_FLASH_ATTN_VERSION',
 'VLLM_TORCH_PROFILER_DIR','VLLM_TORCH_PROFILER_RECORD_SHAPES','VLLM_TORCH_PROFILER_WITH_STACK',
 'PYTHONDONTWRITEBYTECODE','PYTHONPATH','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY',
 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_MPS_CLIENT_PRIORITY','CUDA_DEVICE_MAX_CONNECTIONS')


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def static_environment():
    packages={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}
    executable=Path(sys.executable).resolve()
    return dict(python=sys.version,python_executable=sys.executable,python_binary_sha256=file_hash(executable),
        platform=platform.platform(),machine=platform.machine(),system=platform.system(),
        packages=packages,package_versions_sha256=digest(packages),
        selected_environment={k:os.environ[k] for k in ENV_KEYS if k in os.environ},
        environment_capture_policy='explicit runtime-only allowlist; no credentials or full environment dump',
        source_manifest=runtime_source_manifest())


def torch_state(torch):
    cuda=bool(torch.cuda.is_available());devices=[]
    if cuda:
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            devices.append(dict(logical_index=i,name=p.name,uuid=str(getattr(p,'uuid','unknown')),
                total_memory_bytes=p.total_memory,compute_capability=[p.major,p.minor]))
    return dict(torch_version=str(torch.__version__),cuda_build=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        cuda_available=cuda,devices=devices,compiled_cuda_arches=torch.cuda.get_arch_list() if cuda else [],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        sdpa_enabled=dict(flash=torch.backends.cuda.flash_sdp_enabled(),memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled()),
        enabled_backends_are_not_observed_kernels=True)


def model_state(model):
    counts=Counter();elements=Counter();devices=Counter();attention={};compiled=[]
    for p in model.parameters():counts[str(p.dtype)]+=1;elements[str(p.dtype)]+=p.numel();devices[str(p.device)]+=1
    for name,module in model.named_modules():
        if hasattr(module,'_orig_mod'):compiled.append(name)
        config=getattr(module,'config',None);backend=getattr(config,'_attn_implementation',None)
        if backend is not None:attention[name]=str(backend)
    return dict(model_class=type(model).__module__+'.'+type(model).__qualname__,parameter_tensor_counts_by_dtype=dict(counts),
        parameter_elements_by_dtype=dict(elements),parameter_devices=dict(devices),attention_config_values=attention,
        compiled_module_markers=compiled,cuda_graph_execution=None,complete_attention_kernel_identity=None,
        note='Loaded model attributes only; absence of markers does not prove absence of compilation/graphs')


def command_evidence(argv,timeout=8):
    if not shutil.which(argv[0]):return dict(status='unavailable',reason='executable_not_found')
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout,check=False)
        return dict(status='ok' if r.returncode==0 else 'failed',returncode=r.returncode,stdout=r.stdout[:65536],stderr=r.stderr[:8192],
                    output_truncated=len(r.stdout)>65536 or len(r.stderr)>8192)
    except subprocess.TimeoutExpired:return dict(status='timeout')


def collect_environment(*,include_torch=False):
    result=static_environment()
    result['tools']={
        'nvidia_smi':command_evidence(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader,nounits']),
        'nvcc':command_evidence(['nvcc','--version']), 'nsys':command_evidence(['nsys','--version'])}
    if include_torch:
        try:
            import torch
            result['torch']=torch_state(torch)
        except ImportError:result['torch']={'status':'not_installed'}
    else:result['torch']={'status':'not_requested'}
    result.update(kernel_trace_collected=False,performance_benchmark=False)
    return result


def summarize_trace(path,max_bytes=128*1024*1024):
    path=Path(path)
    if not 1024<=max_bytes<=512*1024*1024:raise ValueError('Trace parser byte cap outside supported bounds')
    if path.stat().st_size>max_bytes:raise ValueError('Trace file exceeds parser byte cap')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as f:data=f.read(max_bytes+1)
    if len(data)>max_bytes:raise ValueError('Decompressed trace exceeds parser byte cap')
    def nonfinite(x):raise ValueError('Nonfinite trace constant')
    obj=json.loads(data,parse_constant=nonfinite);events=obj.get('traceEvents') if isinstance(obj,dict) else obj
    if not isinstance(events,list):raise ValueError('Chrome traceEvents list required')
    kernels=Counter();operators=Counter();durations=Counter();processes=set()
    for e in events:
        if not isinstance(e,dict):raise ValueError('Invalid trace event')
        if e.get('ph')!='X':continue
        name=e.get('name');duration=e.get('dur',0)
        if not isinstance(name,str) or type(duration) not in (int,float) or not math.isfinite(duration) or duration<0:raise ValueError('Invalid complete event')
        cats={s.strip().lower() for s in str(e.get('cat','')).split(',')}
        if 'kernel' in cats:
            kernels[name]+=1;durations[name]+=duration;processes.add(str(e.get('pid','unknown')))
        elif 'cpu_op' in cats:operators[name]+=1
    return dict(trace_sha256=file_hash(path),compressed_bytes=path.stat().st_size,decoded_bytes=len(data),events=len(events),
        gpu_kernel_events=sum(kernels.values()),gpu_kernel_events_present=bool(kernels),kernel_names=dict(kernels),
        kernel_duration_sum_us=dict(durations),gpu_event_process_ids=sorted(processes),cpu_operator_names=dict(operators),
        duration_scope='Sum of recorded events, may overlap; not end-to-end latency or GPU utilization',
        optimization_equivalence_verified=False,complete_kernel_coverage_verified=False)


def capture_torch(call,path,*,cuda=False):
    import torch
    if cuda and not torch.cuda.is_available():raise ValueError('CUDA trace requested without CUDA; no CPU fallback')
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if cuda:activities.append(torch.profiler.ProfilerActivity.CUDA);torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=False,with_stack=False) as profiler:
        with torch.profiler.record_function('rosa_diagnostic_inference'):
            value=call()
        if cuda:torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return value,summarize_trace(path)


async def capture_vllm(engine,call,directory,*,flush_timeout_s=120):
    """Start/stop native vLLM worker profilers; never treat its CPU frontend as CUDA proof."""
    import asyncio
    directory=Path(directory)
    if any(directory.iterdir()):raise ValueError('vLLM trace directory must start empty')
    started=False
    try:
        await asyncio.wait_for(engine.start_profile(),flush_timeout_s);started=True
        value=await call()
    finally:
        # Even a partially successful start must be stopped before shutdown.
        await asyncio.wait_for(engine.stop_profile(),flush_timeout_s)
    traces={str(p.relative_to(directory)):summarize_trace(p) for p in sorted(directory.rglob('*')) if p.is_file() and
            (p.name.endswith('.json') or p.name.endswith('.json.gz'))}
    if not traces:raise ValueError('vLLM profiler produced no readable traces')
    return value,traces
