"""Optional synchronized per-batch CUDA diagnostics, never device utilization.

Peak counters belong to this dedicated worker's PyTorch allocator. Call only
from its serialized synchronous inference path; vLLM engine subprocesses are
not measured here. Measurements change synchronization and therefore the run
binding. Unavailable values stay null, including on CPU.
"""
from collections import Counter
import math

PROTOCOL='torch-events-allocator-v1'


def contract(enabled, *, gpu, asynchronous):
    if type(enabled) is not bool:raise ValueError('cuda_measurement must be boolean')
    reason='cpu_execution' if not gpu else 'async_engine_not_in_worker_process' if asynchronous else 'disabled' if not enabled else None
    return dict(protocol=PROTOCOL,enabled=enabled,unavailable_reason=reason,
        synchronization='device before and after invocation' if reason is None else 'none',
        time_scope='current CUDA stream interval; includes launch gaps; other streams only through dependencies',
        memory_scope='dedicated worker process PyTorch allocator; includes resident model; not device-wide or per-request',
        peak_reset_scope='each serialized backend batch' if reason is None else 'none')


def unavailable(reason):
    return dict(protocol=PROTOCOL,status='unavailable',reason=reason,cuda_elapsed_ms=None,
        allocated_before_bytes=None,reserved_before_bytes=None,peak_allocated_bytes=None,peak_reserved_bytes=None)


def invoke(call, protocol, *, torch_module=None):
    """Return value, evidence and exception without hiding failed inference."""
    sample=unavailable(protocol['unavailable_reason']);value=None;error=None;ready=False
    if protocol['unavailable_reason'] is None:
        try:
            if torch_module is None:
                import torch as torch_module
            cuda=torch_module.cuda
            if not cuda.is_available() or cuda.device_count()!=1:raise ValueError('Expected one visible CUDA device')
            cuda.synchronize();allocated=cuda.memory_allocated();reserved=cuda.memory_reserved()
            cuda.reset_peak_memory_stats()
            start=cuda.Event(enable_timing=True);end=cuda.Event(enable_timing=True)
            start.record();ready=True
        except Exception as exc:sample=unavailable('measurement_setup_error:'+type(exc).__name__)
    try:value=call()
    except Exception as exc:error=exc
    finally:
        if ready:
            try:
                end.record();cuda.synchronize()
                sample=dict(protocol=PROTOCOL,status='measured',reason=None,cuda_elapsed_ms=float(start.elapsed_time(end)),
                    allocated_before_bytes=int(allocated),reserved_before_bytes=int(reserved),
                    peak_allocated_bytes=int(cuda.max_memory_allocated()),peak_reserved_bytes=int(cuda.max_memory_reserved()))
                validate(sample)
            except Exception as exc:sample=unavailable('measurement_finish_error:'+type(exc).__name__)
    sample['inference_succeeded']=error is None
    return value,sample,error


def validate(sample):
    if sample.get('protocol')!=PROTOCOL or sample.get('status') not in ('measured','unavailable'):raise ValueError('Invalid CUDA measurement protocol')
    keys=('cuda_elapsed_ms','allocated_before_bytes','reserved_before_bytes','peak_allocated_bytes','peak_reserved_bytes')
    if sample['status']=='unavailable':
        if not isinstance(sample.get('reason'),str) or not sample['reason'] or any(sample.get(k) is not None for k in keys):raise ValueError('Unavailable CUDA measurement contains values')
    else:
        if sample.get('reason') is not None or any(type(sample.get(k)) not in (int,float) or not math.isfinite(sample[k]) or sample[k]<0 for k in keys):raise ValueError('Invalid measured CUDA values')
        if any(type(sample[k]) is not int for k in keys[1:]) or sample['peak_allocated_bytes']<sample['allocated_before_bytes'] or sample['peak_reserved_bytes']<sample['reserved_before_bytes']:raise ValueError('Invalid allocator peak')
    return sample


def summarize(rows):
    """Deduplicate copied batch evidence; never sum overlapping GPU time."""
    batches={};missing=0
    for row in rows:
        sample=(row.get('backend_evidence') or {}).get('gpu_measurement')
        if sample is None:missing+=1;continue
        validate(sample);bid=row.get('batch_id')
        if not bid or sample.get('batch_id')!=bid or sample.get('actual_batch_size')!=row.get('actual_batch_size'):raise ValueError('CUDA measurement batch binding differs')
        if bid in batches and batches[bid]!=sample:raise ValueError('Inconsistent measurement within batch')
        batches[bid]=sample
    measured=[s for s in batches.values() if s['status']=='measured']
    return dict(protocol=PROTOCOL,unique_batches=len(batches),measured_batches=len(measured),records_missing_evidence=missing,
        unavailable_reasons=dict(Counter(s['reason'] for s in batches.values() if s['status']=='unavailable')),
        cuda_elapsed_ms_mean_per_batch=sum(s['cuda_elapsed_ms'] for s in measured)/len(measured) if measured else None,
        peak_allocated_bytes=max((s['peak_allocated_bytes'] for s in measured),default=None),
        peak_reserved_bytes=max((s['peak_reserved_bytes'] for s in measured),default=None),
        utilization_claim=False,request_latency_claim=False)
