"""Environment inventory and isolated, supervised kernel diagnostics (no Pod management)."""
import argparse,asyncio,importlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.contracts import load_json
from rosa_repro.execution_plan import digest,runtime_source_manifest,check_gpu_device
from rosa_repro.runtime_attestation import collect_environment,capture_torch,capture_vllm,summarize_trace,torch_state,file_hash
from rosa_repro.regression import write_json
from rosa_repro.profile_runner import seal,verify_evidence


def cpu_trace(output):
    import torch
    torch.manual_seed(42)
    q=torch.randn(1,2,16,8)
    def call():return torch.nn.functional.scaled_dot_product_attention(q,q,q)
    call();value,trace=capture_torch(call,output/'cpu-trace.json')
    return dict(status='complete',kind='cpu_torch_self_test',gpu_execution=False,pretrained_weights_loaded=False,
        output_shape=list(value.shape),traces={'cpu-trace.json':trace},gpu_kernel_events_present=False,
        performance_benchmark=False,optimization_equivalence_verified=False)


async def gpu_trace(a,out):
    from rosa_repro.profile_protocol import validate_probe_config,probe_plan,load_corpus
    from rosa_repro.component_runtime import ComponentCore
    from rosa_repro.paired import digest_value
    c=validate_probe_config(load_json(a.config))
    if c['mode']!='gpu':raise ValueError('GPU diagnostic requires a resolved GPU configuration')
    py=c['binding'].get('runtime_env',{}).get('py_executable')
    if py and os.path.abspath(py)!=os.path.abspath(sys.executable):raise ValueError('Run using the declared worker Python, not the driver environment')
    corpus,samples=load_corpus(a.corpus,a.root,'evaluation','gpu')
    if not 0<=a.sample<len(samples):raise ValueError('Sample index out of range')
    cpath=out/'condition.json';write_json(cpath,dict(config=c,config_sha256=digest(c),corpus_id=corpus['corpus_id'],
        observation_sha256=samples[a.sample]['sha256'],split='evaluation',batch_size=c['batch_size'],warmup_batches=1,traced_batches=1,
        scope='isolated backend/core path; excludes Ray scheduling, network and live simulator actions'))
    is_vllm=c['binding']['factory']=='rosa_repro.vlm_backends:vllm_vlm_backend'
    trace_dir=out/'traces';trace_dir.mkdir()
    if is_vllm:
        os.environ['VLLM_TORCH_PROFILER_DIR']=str(trace_dir)
        os.environ['VLLM_TORCH_PROFILER_WITH_STACK']='0'
        os.environ['VLLM_TORCH_PROFILER_RECORD_SHAPES']='1'
    import torch
    state=torch_state(torch)
    if len(state['devices'])!=1:raise ValueError('Isolate exactly one visible GPU for diagnostic; no CPU fallback')
    device=state['devices'][0];plan=probe_plan(c);w=plan['workers'][0]
    check_gpu_device(w,['0'],dict(name=device['name'],memory_mib=device['total_memory_bytes']/1024**2),cuda_available=True,visible_count=1)
    core=ComponentCore(w,plan['plan_id']);records=[]
    async def one(batch_id):
        tickets=[]
        for i in range(c['batch_size']):
            rid=w['members'][i];ack=core.reset('kernel',rid,batch_id)
            m=dict(plan_id=plan['plan_id'],worker_id=w['worker_id'],run_id='kernel',robot_id=rid,episode_id=batch_id,
                request_id=batch_id+'-'+str(i),attempt_id=batch_id+'-'+str(i),component_id=c['component_id'],observation_seq=0,generation=ack['generation'])
            payload=dict(observation=samples[a.sample]['observation'],**c['payload_overlay']);tickets.append(core.accept(m,payload))
        try:
            responses=await core.run_batch(tickets)
            for response in responses:
                output=response.pop('output',None);response['output_sha256']=digest_value(output);records.append(response)
            if any(r['status']!='ok' for r in responses):raise ValueError('Model generation/parser failed; inspect raw diagnostic evidence')
            return responses
        finally:
            for t in tickets:core.finish(t)
    result=dict(status='failed',kind='gpu_backend_kernel_diagnostic',config_sha256=digest(c),backend=core.description,
        source_manifest=runtime_source_manifest(),gpu_execution=True,pretrained_weights_loaded=True,
        performance_benchmark=False,optimization_equivalence_verified=False,actions_executed=0,observed_torch_state=state)
    try:
        await one('warmup')
        if is_vllm:
            _,traces=await capture_vllm(core.backend.engine,lambda:one('trace'),trace_dir,flush_timeout_s=a.flush_timeout)
            result['traces']={'traces/'+k:v for k,v in traces.items()}
        else:
            # Core's synchronous backend runs on another thread. Profile directly
            # in this thread so torch captures the thread executing model ops.
            payload=dict(observation=samples[a.sample]['observation'],**c['payload_overlay'])
            outputs,trace=capture_torch(lambda:core.backend.predict_batch([payload for _ in range(c['batch_size'])]),trace_dir/'torch.json',cuda=True)
            from rosa_repro.component_runtime import ComponentResult,validate_output
            for output in outputs:
                if isinstance(output,ComponentResult):
                    records.append(dict(phase='trace',backend_evidence=output.evidence,error_type=output.error_type,output_sha256=digest_value(output.output)))
                    if output.error_type:raise ValueError('Traced generation was not accepted')
                    output=output.output
                validate_output(c['role'],output)
                if c['role']=='s1':records.append(dict(phase='trace',output_sha256=digest_value(output)))
            if len(outputs)!=c['batch_size']:raise ValueError('Traced batch output count mismatch')
            result['traces']={'traces/torch.json':trace}
        result['gpu_kernel_events_present']=any(t['gpu_kernel_events_present'] for t in result['traces'].values())
        result['status']='complete' if result['gpu_kernel_events_present'] else 'incomplete_no_cuda_kernel_events'
    except Exception as exc:result['error']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if is_vllm:
            try:core.backend.engine.shutdown()
            except Exception as exc:result['shutdown_error']=type(exc).__name__;result['status']='failed'
        with (out/'requests.jsonl').open('x') as f:
            for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
        result['environment_after']=collect_environment(include_torch=True)
    return result


def child(a):
    out=a.output.resolve()
    write_json(out/'source_manifest.json',runtime_source_manifest());write_json(out/'environment-before.json',collect_environment(include_torch=True))
    result=cpu_trace(out) if a.cpu_self_test else asyncio.run(gpu_trace(a,out))
    if a.cpu_self_test:
        write_json(out/'condition.json',dict(kind='synthetic_cpu_tensor_self_test'));(out/'requests.jsonl').write_text('')
    write_json(out/'diagnostic.json',result)
    return 0 if result['status']=='complete' else 2


def supervise(a):
    if not 10<=a.timeout<=3600:raise ValueError('Diagnostic wall-clock timeout must be between 10 and 3600 seconds')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,str(Path(__file__).resolve()),'probe','--child','--output',str(out),'--flush-timeout',str(a.flush_timeout)]
    if a.cpu_self_test:argv+=['--cpu-self-test']
    else:
        if not all((a.config,a.corpus,a.root)):raise ValueError('GPU probe requires config/corpus/root')
        argv+=['--config',str(a.config.resolve()),'--corpus',str(a.corpus.resolve()),'--root',str(a.root.resolve()),'--sample',str(a.sample)]
    started=time.monotonic();timed_out=False;interrupted=False
    with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
        proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=proc.wait(timeout=a.timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            timed_out=isinstance(exc,subprocess.TimeoutExpired);interrupted=not timed_out
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=proc.wait(timeout=5)
    write_json(out/'execution.json',dict(exit_code=code,timed_out=timed_out,interrupted=interrupted,elapsed_s=time.monotonic()-started,wall_budget_s=a.timeout,
        budget_includes='imports, weight verification, model loading, warmup, trace, flush and shutdown; timeout teardown may add 10 seconds',
        gpu_requested=not a.cpu_self_test,pod_operations_performed=False,process_group_supervised=True))
    result=load_json(out/'diagnostic.json') if (out/'diagnostic.json').exists() else dict(status='failed',
        gpu_kernel_events_present=False,optimization_equivalence_verified=False)
    result.update(process_exit_code=code,process_timed_out=timed_out,process_interrupted=interrupted)
    if timed_out or interrupted or code!=0:result['status']='timeout' if timed_out else 'interrupted' if interrupted else 'failed'
    write_json(out/'result.json',result)
    seal(out);return 0 if code==0 and not timed_out and not interrupted else 2


def compare(diagnostic,profile):
    verify_evidence(diagnostic);verify_evidence(profile)
    d=load_json(Path(diagnostic)/'condition.json');p=load_json(Path(profile)/'condition.json');r=load_json(Path(diagnostic)/'result.json')
    match=d.get('config')==p.get('config') and d.get('config') is not None
    source=load_json(Path(diagnostic)/'source_manifest.json')==load_json(Path(profile)/'source_manifest.json')
    return dict(config_matches=match,source_matches=source,diagnostic_completed=r['status']=='complete',
        cuda_events_present=r.get('gpu_kernel_events_present',False),automatic_profile_acceptance=False,
        requires_review=['worker Python/package versions and GPU identity','actual operator/kernel names and dtype','input/batch representativeness'],
        scope='Configuration/source association only; diagnostic timing is excluded from performance profiles')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('collect');q.add_argument('--torch',action='store_true');q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('trace-summary');q.add_argument('trace',type=Path);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('compare');q.add_argument('--diagnostic',type=Path,required=True);q.add_argument('--profile',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('probe');q.add_argument('--cpu-self-test',action='store_true');q.add_argument('--config',type=Path);q.add_argument('--corpus',type=Path);q.add_argument('--root',type=Path)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--sample',type=int,default=0);q.add_argument('--timeout',type=int,default=600)
    q.add_argument('--flush-timeout',type=int,default=120);q.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.command=='collect':write_json(a.output,collect_environment(include_torch=a.torch))
    elif a.command=='trace-summary':write_json(a.output,summarize_trace(a.trace))
    elif a.command=='compare':write_json(a.output,compare(a.diagnostic,a.profile))
    else:return child(a) if a.child else supervise(a)
    return 0


if __name__=='__main__':raise SystemExit(main())
