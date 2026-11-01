import asyncio,json
from types import SimpleNamespace
import pytest
from rosa_repro.gpu_measurement import contract,invoke,summarize,validate
from test_r05 import run,Transport,suite


class Cuda:
 def __init__(self,fail=None):self.calls=[];self.fail=fail
 def is_available(self):return True
 def device_count(self):return 1
 def synchronize(self):
  self.calls.append('sync')
  if self.fail=='finish' and len(self.calls)>3:raise RuntimeError('CUDA failure')
 def memory_allocated(self):return 10
 def memory_reserved(self):return 30
 def reset_peak_memory_stats(self):
  self.calls.append('reset')
  if self.fail=='setup':raise RuntimeError('reset failure')
 def max_memory_allocated(self):return 20
 def max_memory_reserved(self):return 40
 def Event(self,enable_timing):
  assert enable_timing
  return SimpleNamespace(record=lambda:self.calls.append('event'),elapsed_time=lambda end:2.5)


def test_cuda_scope_order_peak_and_deduplication():
 cuda=Cuda();p=contract(True,gpu=True,asynchronous=False)
 value,s,error=invoke(lambda:cuda.calls.append('model') or 42,p,torch_module=SimpleNamespace(cuda=cuda))
 assert value==42 and error is None and cuda.calls==['sync','reset','event','model','event','sync']
 assert s['cuda_elapsed_ms']==2.5 and s['peak_allocated_bytes']==20
 s.update(batch_id='b',actual_batch_size=2)
 row=dict(batch_id='b',actual_batch_size=2,backend_evidence={'gpu_measurement':s})
 summary=summarize([row,row]);assert summary['unique_batches']==summary['measured_batches']==1
 assert not summary['utilization_claim'] and not summary['request_latency_claim']


@pytest.mark.parametrize('failure',['setup','finish','inference'])
def test_measurement_and_inference_failures_are_distinct(failure):
 cuda=Cuda(failure)
 def call():
  if failure=='inference':raise LookupError('model failure')
  return 1
 v,s,e=invoke(call,contract(True,gpu=True,asynchronous=False),torch_module=SimpleNamespace(cuda=cuda))
 assert (isinstance(e,LookupError) and s['status']=='measured') if failure=='inference' else (v==1 and e is None and s['status']=='unavailable')
 assert (s['cuda_elapsed_ms'] is None)==(failure!='inference')


@pytest.mark.parametrize('gpu,asynchronous,enabled,reason',[(False,False,True,'cpu_execution'),(True,True,True,'async_engine_not_in_worker_process'),(True,False,False,'disabled')])
def test_unavailable_never_fabricates_cuda(gpu,asynchronous,enabled,reason):
 v,s,e=invoke(lambda:7,contract(enabled,gpu=gpu,asynchronous=asynchronous))
 assert v==7 and s['reason']==reason and s['peak_allocated_bytes'] is None
 validate(s)
 s['cuda_elapsed_ms']=0
 with pytest.raises(ValueError):validate(s)


def test_cpu_raw_records_and_measurement_reaudit(tmp_path):
 from rosa_repro.r05_runtime import reaudit,seal
 r,out,_,_=run(tmp_path)
 assert r['metrics']['measurement']['gpu_measurement']['unavailable_reasons']['cpu_execution']>0
 rows=list(map(json.loads,(out/'events.jsonl').read_text().splitlines()))
 row=next(e for e in rows if e['kind']=='request_end')
 row['backend_evidence']['gpu_measurement']['batch_id']='forged'
 (out/'events.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n');(out/'evidence.json').unlink();seal(out)
 with pytest.raises(ValueError,match='measurement batch'):reaudit(out)


def test_periodic_telemetry_and_failures_remain_in_evidence(tmp_path):
 class FailingTelemetry(Transport):
  async def telemetry(self,w):raise RuntimeError('device utility unavailable')
 s,root=suite(tmp_path);s['protocol']['duration_s']=1.2
 from rosa_repro.r05_protocol import prepare
 from rosa_repro.paper_conditions import generate
 s=prepare(generate(),s['config'],s['corpus'],stage='pilot',groups=['k4-a50'],protocol=s['protocol'])
 r,out,_,_=run(tmp_path,s=s,root=root,transport_type=FailingTelemetry)
 assert r['status']=='complete' and r['telemetry']['periodic_samples']>=1
 assert r['telemetry']['kinds']=={'unavailable':r['telemetry']['samples']}
 assert all(e['sample']['error_type']=='RuntimeError' for e in map(json.loads,(out/'telemetry.jsonl').read_text().splitlines()))


def test_instrumentation_mode_binds_adoption_signature(tmp_path):
 from test_profile_adoption import fixture,read
 from rosa_repro.profile_adoption import runtime_signature
 fixture(tmp_path);r=read(tmp_path/'kernel/result.json');b=r['backend'];env=r['environment_after']
 a=runtime_signature(env,b);b['gpu_measurement']=contract(True,gpu=True,asynchronous=False)
 assert runtime_signature(env,b)!=a
