import asyncio,json,hashlib
from copy import deepcopy
import pytest
from rosa_repro.r05_protocol import template,config,timing,prepare,validate_suite,plan_for,validate_plan
from rosa_repro.r05_runtime import run_trial,reaudit,seal
from rosa_repro.r05_workflow import report,check_pilots
from rosa_repro.paper_conditions import generate
from rosa_repro.execution_plan import DUMMY_FACTORY
from rosa_repro.component_runtime import ComponentCore
from test_profile_preparation import corpus


def cpu(**kwargs):
 c=template();c.update(mode='cpu_dummy',model_revision='synthetic',runtime_fingerprint='synthetic',hardware_kind='cpu-fixture',
     binding=dict(factory=DUMMY_FACTORY,kwargs=dict(delay_s=.005)),batch_wait_ms=20,watchdog_s=2,drain_s=2,setup_budget_s=20)
 c.update(kwargs);return c


def suite(tmp_path,*,c=None,group='k4-a50',**timing_kw):
 root,book=corpus(tmp_path);c=c or cpu()
 s=prepare(generate(),c,book,stage='pilot',groups=[group],protocol=timing('cpu_smoke',warmup_s=.12,duration_s=.2,**timing_kw))
 return s,root

class Transport:
 """Use the real backend/session core; coalesce concurrent callers for offline testing."""
 def __init__(self,plan):
  self.plan=plan;self.worker=plan['workers'][0];self.core=ComponentCore(self.worker,plan['plan_id']);self.queue=[];self.task=None
 async def reset(self,w,*args):return self.core.reset(*args)
 async def environment(self,w):return {'kind':'cpu_test'}
 async def telemetry(self,w):return {'kind':'cpu_no_gpu'}
 async def inventory(self):return {self.worker['worker_id']:dict(self.core.stats(),node_id='test',gpu_ids=[])}
 async def predict(self,w,m,payload):
  ticket=self.core.accept(m,payload);future=asyncio.get_running_loop().create_future();self.queue.append((ticket,future))
  if self.task is None or self.task.done():self.task=asyncio.create_task(self.flush())
  return await asyncio.shield(future)
 async def flush(self):
  while self.queue:
   await asyncio.sleep(self.worker['batch_wait_ms']/1000)
   items=self.queue[:self.worker['batch_size']];del self.queue[:len(items)]
   values=await self.core.run_batch([t for t,f in items])
   for (t,f),v in zip(items,values):self.core.finish(t);f.set_result(v)


def run(tmp_path,*,s=None,root=None,batch=4,transport_type=Transport):
 if s is None:s,root=suite(tmp_path)
 cell=next(c for c in s['cells'] if c['batch_size']==batch);out=tmp_path/'trial'
 async def go():
  t=transport_type(plan_for(s['config'],cell))
  r=await run_trial(s,root,cell['id'],t,out)
  if t.task:await t.task
  return r
 return asyncio.run(go()),out,s,cell


def test_no_deadline_plan_accepts_batch_cap_greater_than_robots(tmp_path):
 s,root=suite(tmp_path,group='k1-a50');p=plan_for(s['config'],next(c for c in s['cells'] if c['batch_size']==16))
 assert p['workers'][0]['batch_size']==16 and len(p['workers'][0]['members'])==1
 assert validate_plan(p)==p and validate_suite(s,root)==s and p['latency_slo_ms'] is None
 assert 'deadline_ms' not in json.dumps(p) and 'frequency_cap_hz": null' in json.dumps(p)
 r,out,_,_=run(tmp_path,s=s,root=root,batch=16)
 assert r['status']=='complete' and list(r['metrics']['measurement']['actual_batch_histogram_requests'])==['1']


def test_true_batching_closed_loop_and_no_slo_metrics(tmp_path):
 r,out,s,cell=run(tmp_path)
 assert r['status']=='complete' and reaudit(out)['metrics']==r['metrics']
 m=r['metrics']['measurement'];assert m['completed_virtual_action_chunks']>0 and '4' in m['actual_batch_histogram_requests']
 assert not m['slo_qualification_applied'] and m['latency_slo_ms'] is None and r['physical_actions_executed']==0


def test_valid_response_over_200ms_is_used(tmp_path):
 c=cpu();c['binding']['kwargs']['delay_s']=.24
 root,book=corpus(tmp_path);s=prepare(generate(),c,book,stage='pilot',groups=['k1-a50'],protocol=timing('cpu_smoke',warmup_s=.4,duration_s=.7))
 r,out,_,_=run(tmp_path,s=s,root=root,batch=1)
 assert r['status']=='complete' and r['metrics']['measurement']['latency_ms_successful_response_cohort']['p50']>200
 assert r['metrics']['measurement']['completed_virtual_action_chunks']>0


def test_warmup_shortfall_never_starts_measurement(tmp_path):
 s,root=suite(tmp_path,warmup_min_success=500)
 r,out,_,_=run(tmp_path,s=s,root=root)
 assert r['status']=='warmup_shortfall' and not r['measurement_started'] and r['metrics'] is None
 assert all(e['phase']=='warmup' for e in map(json.loads,(out/'events.jsonl').read_text().splitlines()))


def test_watchdog_failure_drains_late_response_without_action(tmp_path):
 c=cpu(watchdog_s=.02,drain_s=.3);c['binding']['kwargs']['delay_s']=.08
 s,root=suite(tmp_path,c=c,group='k1-a50');r,out,_,_=run(tmp_path,s=s,root=root,batch=1)
 assert r['status']=='failed' and r['drained'] and not r['measurement_started']
 events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
 assert any(e['kind']=='late_response' for e in events) and not any(e['kind']=='action_start' for e in events)


def test_boundary_action_retained_but_not_counted_as_completed_in_window(tmp_path):
 root,book=corpus(tmp_path);s=prepare(generate(),cpu(),book,stage='pilot',groups=['k1-a50'],protocol=timing('cpu_smoke',warmup_s=.1,duration_s=.09))
 r,out,_,_=run(tmp_path,s=s,root=root,batch=1)
 m=r['metrics']['measurement'];assert m['completed_virtual_action_chunks']<=1
 from rosa_repro.r05_runtime import metrics
 rows=[dict(kind='action_end',phase='measurement',time_ns=t,completed=True) for t in (80,110)]
 m=metrics(rows,dict(phase='measurement',start_ns=0,end_ns=100))
 assert m['completed_virtual_action_chunks']==1 and m['cross_boundary_virtual_actions']==1


def test_wrong_robot_response_is_never_applied(tmp_path):
 class Wrong(Transport):
  async def predict(self,*args):r=await super().predict(*args);r['metadata']['robot_id']='other';return r
 s,root=suite(tmp_path,group='k1-a50');r,out,_,_=run(tmp_path,s=s,root=root,batch=1,transport_type=Wrong)
 assert r['status']=='failed' and r['warmup_metrics']['completed_virtual_action_chunks']==0


@pytest.mark.parametrize('change',['fake_deadline','window','action_wait','batch','gate','sample'])
def test_resealed_inconsistent_evidence_rejected(tmp_path,change):
 r,out,_,_=run(tmp_path)
 if change in ('fake_deadline','window'):
  p=out/'result.json';d=json.loads(p.read_text())
  if change=='fake_deadline':d['latency_slo_ms']=200
  else:d['windows'][-1]['end_ns']+=1
  p.write_text(json.dumps(d))
 elif change=='gate':
  p=out/'warmup_gate.json';d=json.loads(p.read_text());d['successful_before_cutoff']+=1;p.write_text(json.dumps(d))
 else:
  p=out/'events.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
  if change=='action_wait':next(e for e in rows if e['kind']=='action_start')['requested_duration_s']=0
  if change=='batch':next(e for e in rows if e['kind']=='request_end')['actual_batch_size']=100
  if change=='sample':next(e for e in rows if e['kind']=='request_start')['input_sha256']='wrong'
  p.write_text('\n'.join(json.dumps(e) for e in rows)+'\n')
 (out/'evidence.json').unlink();seal(out)
 with pytest.raises(ValueError):reaudit(out)


def test_prediction_and_sample_budget_must_be_resolved(tmp_path):
 s,root=suite(tmp_path)
 with pytest.raises(ValueError,match='Resolve every predicted'):prepare(generate(),s['config'],s['corpus'],stage='confirmation',groups=['k4-a50'])
 c=template();c.update(runtime_fingerprint='verified',hardware_kind='GPU');c['binding']['kwargs'].update(repo='/repo',model_path='/model',stateless_verified=True)
 blocked=prepare(generate(),c,dict(s['corpus'],kind='saved_robot_observations'),stage='pilot',groups=['k1-a1000'],protocol=timing('formal'))
 assert not blocked['executable'] and blocked['blocking_reasons'][0]['upper_bound']==119
 with pytest.raises(ValueError):timing('formal',duration_s=10)
 from jsonschema.exceptions import ValidationError
 with pytest.raises(ValidationError):config(dict(cpu(),deadline_ms=1e9))


def test_prospective_missing_cells_and_no_cherrypicked_speedup(tmp_path):
 r,out,s,cell=run(tmp_path)
 mapping={c['id']:'missing-'+c['id'] for c in s['cells']};mapping[cell['id']]='trial'
 report_value=report(s,tmp_path,mapping)
 assert report_value['status']=='incomplete' and len(report_value['rows'])==5
 assert report_value['groups'][0]['paired_speedup'] is None
 with pytest.raises(ValueError):report(s,tmp_path,{c['id']:'trial' for c in s['cells']})


def confirmation(tmp_path):
 from rosa_repro.execution_plan import digest
 from rosa_repro.regression import sha256
 s,root=suite(tmp_path)
 paths=[]
 for cell in s['cells']:
  out=tmp_path/f"b{cell['batch_size']}"
  async def go():
   t=Transport(plan_for(s['config'],cell));return await run_trial(s,root,cell['id'],t,out)
  assert asyncio.run(go())['status']=='complete';paths.append(out.name)
 from rosa_repro.r05_prediction import generate as predict
 prediction=predict(tmp_path,{'k4-a50':paths});text=json.dumps(prediction)
 h=hashlib.sha256(text.encode()).hexdigest();paper=generate(r05_selection={'k4-a50':dict(batch_size=prediction['groups']['k4-a50'],prediction_evidence_sha256=h)})
 confirm=prepare(paper,s['config'],s['corpus'],stage='confirmation',groups=['k4-a50'],protocol=s['protocol'],prediction_artifacts={h:text})
 return confirm,root,{'k4-a50':paths}


def test_confirmation_requires_all_five_pilots_and_exact_observed_runtime(tmp_path):
 s,root,dependencies=confirmation(tmp_path);proof=check_pilots(s,tmp_path,dependencies)
 assert len(s['cells']) in (3,6) and len(proof['k4-a50']['rows'])==5
 with pytest.raises(ValueError):check_pilots(s,tmp_path,{'k4-a50':dependencies['k4-a50'][:-1]})
 with pytest.raises(ValueError):check_pilots(s,tmp_path,{'k4-a50':dependencies['k4-a50']+[dependencies['k4-a50'][0]]})
 bad=deepcopy(s);bad['config']['runtime_fingerprint']='other'
 with pytest.raises(ValueError,match='binding differs'):check_pilots(bad,tmp_path,dependencies)
 cell=s['cells'][0]
 async def go():
  t=Transport(plan_for(s['config'],cell))
  return await run_trial(s,root,cell['id'],t,tmp_path/'wrong-runtime',pilot_runtime={'wrong':'runtime'})
 r=asyncio.run(go());assert r['status']=='failed' and r['metrics'] is None


def test_independent_confirmation_speedup_and_missing_suppression(tmp_path):
 s,root,dependencies=confirmation(tmp_path);check_pilots(s,tmp_path,dependencies);paths={}
 for cell in s['cells']:
  out=tmp_path/cell['id'];paths[cell['id']]=out.name
  async def go():
   t=Transport(plan_for(s['config'],cell));return await run_trial(s,root,cell['id'],t,out)
  assert asyncio.run(go())['status']=='complete'
 r=report(s,tmp_path,paths);assert r['status']=='complete'
 assert r['groups'][0]['paired_speedup']['summary']['n']==3 and len(r['groups'][0]['paired_speedup']['pairs'])==3
 paths[s['cells'][0]['id']]='missing';r=report(s,tmp_path,paths)
 assert r['status']=='incomplete' and r['groups'][0]['paired_speedup'] is None


def test_forged_prediction_binding_rejected(tmp_path):
 s,root=suite(tmp_path);paper=generate(r05_selection={'k4-a50':dict(batch_size=4,prediction_evidence_sha256='0'*64)})
 with pytest.raises(ValueError,match='Prediction evidence'):prepare(paper,s['config'],s['corpus'],stage='confirmation',groups=['k4-a50'],prediction_artifacts={'0'*64:'{}'})


def test_request_budget_stops_and_all_requests_remain(tmp_path):
 s,root=suite(tmp_path,max_requests=2);r,out,_,_=run(tmp_path,s=s,root=root)
 assert r['status']=='failed'
 events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
 assert sum(e['kind']=='request_start' for e in events)==2 and sum(e['kind']=='request_end' for e in events)==2


def test_latency_statistics_reuse_no_slo_cohort(tmp_path):
 from rosa_repro.statistical_evaluation import latency_report
 r,out,_,_=run(tmp_path)
 a=latency_report(out,kind='r05',block_ns=10000000,resamples=100)
 assert a['latency_slo_ms'] is None and a['groups']['s1']['planned_logged_arrivals']==r['metrics']['measurement']['requests']


def test_cli_dry_run_budget_and_source_binding(tmp_path):
 import importlib.util
 from pathlib import Path
 from types import SimpleNamespace
 from rosa_repro.regression import write_json
 spec=importlib.util.spec_from_file_location('r05_cli_test',Path(__file__).resolve().parents[1]/'scripts/r05_experiments.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 s,root=suite(tmp_path);write_json(tmp_path/'suite.json',s)
 a=SimpleNamespace(suite=tmp_path/'suite.json',corpus_root=root,cell=None,execute=False,output=tmp_path/'runs',
     budget_seconds=1,max_new=1,pilots=None,evidence_root=tmp_path,address=None,cleanup_review=None)
 assert mod.execute(a)['status']=='planned' and not a.output.exists()
 a.execute=True;assert mod.execute(a)['status']=='budget_stop' and not list(a.output.glob('r05-*'))
 bad=deepcopy(s);bad['source_manifest']['sha256']='0'*64
 with pytest.raises(ValueError,match='suite/source'):validate_suite(bad)
