import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from rosa_repro.component_runtime import ComponentCore,AdmissionRejected
from rosa_repro.dummy import observation
from rosa_repro.execution_plan import DUMMY_FACTORY,digest
from rosa_repro.regression import save_observation,write_json
from rosa_repro.profile_protocol import (template,validate_probe_config,probe_plan,validate_probe_plan,build_corpus,load_corpus,
    make_suite,validate_suite,gpu_telemetry,preflight,environment_evidence)
from rosa_repro.profile_runner import run_probe,reaudit,profile_candidate,metrics
from rosa_repro.paired import digest_value


def config(role='safety',**kwargs):
    c=template(role,'groot' if role=='s1' else 'vllm')
    c.update(mode='cpu_dummy',hardware_kind='cpu-fixture',model_revision='synthetic',runtime_fingerprint='synthetic',
        binding=dict(factory=DUMMY_FACTORY,kwargs=dict(delay_s=.002)),streams=2,request_timeout_s=.1,drain_timeout_s=.2)
    c.update(kwargs);return c


def corpus(tmp_path):
    root=tmp_path/'observations'
    for i in range(4):
        for arm in ('official','shared'):
            save_observation(root/str(i)/arm,observation(i+1),dict(task='synthetic-task',case={'state_sha256':digest_value([i])},execution_kind='cpu_dummy'))
    return root,build_corpus(root,kind='synthetic_cpu')


class Transport:
    def __init__(self,p):self.plan=p;self.w=p['workers'][0];self.core=ComponentCore(self.w,p['plan_id']);self.tasks=set();self.lock=asyncio.Lock()
    async def reset(self,w,*args):return self.core.reset(*args)
    async def predict(self,w,m,payload):
        try:t=self.core.accept(m,payload)
        except AdmissionRejected:return dict(self.core.response({'meta':m}),status='rejected')
        async def execute():
            try:
                async with self.lock:return (await self.core.run_batch([t]))[0]
            finally:self.core.finish(t)
        task=asyncio.create_task(execute());self.tasks.add(task);task.add_done_callback(self.tasks.discard)
        return await asyncio.shield(task)
    async def inventory(self):return {self.w['worker_id']:dict(self.core.stats(),node_id='cpu-test',gpu_ids=[],gpu_properties=None)}
    async def environment(self,w):return dict(kind='synthetic_environment')
    async def telemetry(self,w):return dict(kind='cpu_no_gpu')


def run(tmp_path,*,c=None,stage='profile',rate=50):
    c=c or config();root,book=corpus(tmp_path)
    suite=make_suite(c,book,stage=stage,phase='cpu_smoke',rates=[rate] if stage=='profile' else [])
    _,samples=load_corpus(book,root,suite['split'],c['mode']);p=probe_plan(c);out=tmp_path/'trial'
    async def go():
        t=Transport(p);r=await run_probe(p,t,samples,suite['cases'][0],out,corpus_id=book['corpus_id'],split=suite['split'])
        if t.tasks:await asyncio.gather(*t.tasks)
        return r
    return asyncio.run(go()),out


@pytest.mark.parametrize('role,engine',[('s1','groot'),('s2','vllm'),('safety','transformers'),('monitor','vllm')])
def test_templates_are_unresolved_and_cannot_deploy(role,engine):
    c=template(role,engine);validate_probe_config(c,ready=False)
    with pytest.raises(ValueError,match='unresolved'):probe_plan(c)


@pytest.mark.parametrize('change',['fingerprint','batch','timeout','relative_path'])
def test_resolved_gpu_config_enforces_backend_contract(change):
    c=template('s2');c['hardware_kind']='GPU';c['binding']['kwargs']['model_path']='/workspace/model'
    if change=='fingerprint':c['runtime_fingerprint']='wrong'
    if change=='batch':c['batch_size']=2
    if change=='timeout':c['request_timeout_s']=1
    if change=='relative_path':c['binding']['kwargs']['model_path']='relative'
    with pytest.raises(ValueError):validate_probe_config(c)


def test_probe_has_no_fabricated_schedule_profile_and_digest_is_checked():
    p=probe_plan(config());validate_probe_plan(p)
    assert p['workers'][0]['profile_source']=='unmeasured' and 'R07' in p['purpose']
    p['workers'][0]['resources']['num_gpus']=1
    with pytest.raises(ValueError):validate_probe_plan(p)


def test_corpus_split_keeps_arms_together_and_checks_file_content(tmp_path):
    root,c=corpus(tmp_path);_,xs=load_corpus(c,root,'profile','cpu_dummy')
    assert len(xs)==4 and len({r['group_id'] for r in c['rows']})==4
    for g in {r['group_id'] for r in c['rows']}:assert len({r['split'] for r in c['rows'] if r['group_id']==g})==1
    row=c['rows'][0];(root/row['directory']/'arrays.npz').write_bytes(b'broken')
    with pytest.raises(ValueError):load_corpus(c,root,'profile','cpu_dummy')


@pytest.mark.parametrize('bad',['leakage','traversal','synthetic_gpu'])
def test_corpus_rejects_leakage_paths_and_gpu_relabelling(tmp_path,bad):
    root,c=corpus(tmp_path)
    if bad=='leakage':c['rows'][1]['split']='evaluation' if c['rows'][0]['split']=='profile' else 'profile'
    if bad=='traversal':c['rows'][0]['directory']='../outside'
    c.pop('corpus_id');c['corpus_id']=digest(c)
    with pytest.raises(ValueError):load_corpus(c,root,'profile','gpu' if bad=='synthetic_gpu' else 'cpu_dummy')


@pytest.mark.parametrize('rates',[[],[0],[float('nan')],[1,1],[-1]])
def test_profile_requires_resolved_loads(tmp_path,rates):
    _,book=corpus(tmp_path)
    with pytest.raises(ValueError):make_suite(config(),book,stage='profile',phase='cpu_smoke',rates=rates)


def test_suite_is_source_bound_and_cpu_cannot_become_formal_GPU(tmp_path):
    _,book=corpus(tmp_path);s=make_suite(config(),book,stage='profile',phase='cpu_smoke',rates=[1,2],repetitions=3)
    validate_suite(s);assert len(s['cases'])==6
    s['cases'][0]['protocol']['measure_min_s']=0
    with pytest.raises(ValueError):validate_suite(s)
    c=template('s2');c['hardware_kind']='GPU';c['binding']['kwargs']['model_path']='/workspace/model'
    with pytest.raises(ValueError):make_suite(c,book,stage='smoke',phase='cpu_smoke')


def test_successful_probe_logs_can_be_recomputed_but_CPU_export_is_blocked(tmp_path):
    r,out=run(tmp_path)
    assert r['status']=='complete' and r['drained'] and r['metrics']['successful']==r['metrics']['arrivals']
    assert reaudit(out)==r['metrics'] and r['metrics']['actions_executed']==0
    candidate=profile_candidate([out]);assert candidate['profile'] is None
    assert 'cpu_results_are_not_GPU_profiles' in candidate['blocking_reasons']
    assert not candidate['automatic_scheduler_import_allowed']
    with pytest.raises(ValueError,match='Duplicate'):profile_candidate([out,out])
    with (out/'requests.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='modified'):reaudit(out)


def test_timeouts_keep_raw_denominator_and_do_not_turn_into_late_success(tmp_path):
    c=config(request_timeout_s=.005);c['binding']['kwargs']['delay_s']=.02
    r,out=run(tmp_path,c=c)
    assert r['status']=='warmup_shortfall' and r['drained']
    rows=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
    assert any(x['status']=='timeout' for x in rows) and all(x['status']!='ok' for x in rows)
    assert not r.get('metrics')


def test_overload_is_recorded_as_unsent_arrivals(tmp_path):
    c=config();c['binding']['kwargs']['delay_s']=.01
    r,out=run(tmp_path,c=c,rate=500)
    assert r['metrics']['status_counts']['not_sent']>0
    assert r['metrics']['arrivals']>r['metrics']['successful']
    assert reaudit(out)==r['metrics']


def test_low_offered_rate_obeys_phase_upper_bound(tmp_path):
    r,out=run(tmp_path,rate=.001)
    assert r['status']=='warmup_shortfall' and r['warmup']['duration_s']<.65


def test_delayed_wakeup_does_not_backdate_ready_warmup(tmp_path,monkeypatch):
    import time
    import rosa_repro.profile_runner as runner
    clock=SimpleNamespace(ns=0,stalled=False)
    real_sleep=asyncio.sleep
    async def delayed_sleep(delay):
        # Let inference complete, then simulate an event-loop wakeup that skips
        # the 20ms arrival and crosses the 30ms minimum duration.
        await real_sleep(.01)
        advance=round(delay*1e9)
        if not clock.stalled and clock.ns>=10_000_000:
            advance+=20_000_000;clock.stalled=True
        clock.ns+=advance
    monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic_ns=lambda:clock.ns,monotonic=time.monotonic))
    monkeypatch.setattr(runner,'asyncio',SimpleNamespace(**{**vars(asyncio),'sleep':delayed_sleep}))
    r,out=run(tmp_path,rate=100)
    assert clock.stalled and r['warmup']['requirements_met']
    assert r['warmup']['duration_s']>=.03 and r['warmup']['successful_before_cutoff']>=2
    rows=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
    missed=[x for x in rows if x['phase']=='warmup' and x['status']=='not_sent']
    assert missed and all(x['reason']=='phase_cutoff_before_dispatch' for x in missed)
    assert r['status']=='complete' and reaudit(out)==r['metrics']


def test_metrics_include_unsent_and_post_cutoff_responses():
    rows=[dict(request_id='a',scheduled_ns=0,sent_ns=10,finished_ns=150,status='ok',actual_batch_size=1),
          dict(request_id='b',scheduled_ns=20,sent_ns=20,finished_ns=20,status='not_sent')]
    m=metrics(rows,{'start_ns':0,'end_ns':100},.00005)
    assert m['arrivals']==2 and m['completed_after_cutoff']==1 and m['deadline_meet_rate_all_arrivals']==0
    assert m['latency_ms_successful_only']['mean']==.00014


def test_s1_evidence_saved_off_loop_and_disk_failure_blocks_trial(tmp_path,monkeypatch):
    import threading
    import numpy as np
    original=np.savez_compressed;threads=[];fail=False
    def save(path,**kwargs):
        if Path(path).name.startswith('action-'):
            threads.append(threading.get_ident())
            if fail:raise OSError('injected full disk')
        return original(path,**kwargs)
    monkeypatch.setattr(np,'savez_compressed',save)
    r,out=run(tmp_path/'good',c=config('s1'))
    assert threads and threading.get_ident() not in threads
    assert r['status']=='complete' and reaudit(out)==r['metrics']
    rows=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
    for row in rows:
        if row['status']=='ok':
            with np.load(out/row['output_file']) as a:assert digest_value(dict(a))==row['output_sha256']
    fail=True
    r,out=run(tmp_path/'bad',c=config('s1'))
    assert r['status']=='failed' and r['error_type']=='EvidencePersistenceFailed'


def test_readonly_cpu_preflight_does_not_import_cuda():
    r=preflight(config());assert r['status']=='passed' and not r['gpu_model_loaded']


def test_gpu_telemetry_is_UUID_bound_and_unavailable_is_explicit(monkeypatch):
    import rosa_repro.profile_protocol as p
    calls=[]
    def cmd(args,**kwargs):calls.append(args);return SimpleNamespace(stdout='GPU-abc, Test GPU, 999, 24000, 1000, 25\n')
    monkeypatch.setattr(p.subprocess,'run',cmd)
    r=gpu_telemetry({'uuid':'GPU-abc'});assert r['used_mib']==1000 and '--id=GPU-abc' in calls[0]
    assert gpu_telemetry({'uuid':'unknown'})['kind']=='unavailable'
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='GPU-other, Test GPU, 999, 24000, 1000, 25\n'))
    assert gpu_telemetry({'uuid':'GPU-abc'})['kind']=='unavailable'


@pytest.mark.parametrize('prefix',['','GPU-'])
def test_torch_physical_uuid_matches_nvidia_identity_without_index_fallback(monkeypatch,prefix):
    import rosa_repro.profile_protocol as p
    physical='2fe1ce1e-2d86-339b-0c13-d1b2b90ff487';calls=[]
    def query(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=f'GPU-{physical}, Test GPU, 999, 97887, 1000, 25\n')
    monkeypatch.setattr(p.subprocess,'run',query)
    r=gpu_telemetry({'uuid':prefix+physical})
    assert r['kind']=='nvidia_smi_sample' and r['uuid']==prefix+physical
    assert r['nvidia_smi_uuid']=='GPU-'+physical and '--id=GPU-'+physical in calls[0]
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='GPU-00000000-0000-0000-0000-000000000000, Test GPU, 999, 97887, 1000, 25\n'))
    assert gpu_telemetry({'uuid':prefix+physical})['kind']=='unavailable'


def test_pilot_hints_and_exact_output_review_are_not_quality_claims(tmp_path):
    from rosa_repro.profile_runner import pilot_rate_hints,compare_probes
    r,out=run(tmp_path,stage='pilot')
    hints=pilot_rate_hints(out)
    assert len(hints['suggested_rates_hz'])==10 and not hints['capacity_certified']
    compare=compare_probes(out,out)
    assert compare['common_inputs']>0 and all(row['exact_output_sets_equal'] for row in compare['rows'])
    assert not compare['semantic_equivalence_verified']


def test_gpu_run_rejected_before_ray_import_or_connection(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/postpod_models.py'
    spec=importlib.util.spec_from_file_location('postpod_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root,book=corpus(tmp_path)
    c=template('s2');c['hardware_kind']='GPU';c['binding']['kwargs']['model_path']='/workspace/model'
    book['kind']='saved_robot_observations';book.pop('corpus_id');book['corpus_id']=digest(book)
    s=make_suite(c,book,stage='smoke',phase='exploratory')
    with pytest.raises(ValueError,match='explicit existing Ray'):
        asyncio.run(module.execute(s,tmp_path/'run',root,address=None))


def formal_fixture(parent,rep):
    # Fabricated GPU metadata only for exporter validation; never executed or released as measurements.
    from rosa_repro.profile_runner import seal
    from rosa_repro.profile_protocol import phase_protocol
    from rosa_repro.execution_plan import runtime_source_manifest
    p=parent/str(rep);p.mkdir();c=template('s2');c['hardware_kind']='Test GPU';c['binding']['kwargs']['model_path']='/workspace/model'
    case=dict(stage='profile',phase='formal',rep=rep,offered_hz=10.,protocol=phase_protocol('formal'))
    from rosa_repro.profiling_budget import formal_timing
    case['timing']=formal_timing(10.,c);case['protocol']=case['timing']['protocol']
    warm=dict(start_ns=0,end_ns=30_000_000_000,duration_s=30.,successful_before_cutoff=300,requirements_met=True,request_limit_hit=False)
    measure=dict(start_ns=40_000_000_000,end_ns=160_000_000_000,duration_s=120.,successful_before_cutoff=1200,requirements_met=True,request_limit_hit=False)
    rows=[]
    for name,w in [('warmup',warm),('measure',measure)]:
        for i in range(w['successful_before_cutoff']):
            tick=w['start_ns']+i*100_000_000
            rows.append(dict(request_id=f'{name}-{i}',phase=name,scheduled_ns=tick,sent_ns=tick,finished_ns=tick+1_000_000,status='ok',actual_batch_size=1))
    (p/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    telemetry=dict(kind='nvidia_smi_sample',uuid='GPU-fake',name='Test GPU',driver='test-only',total_mib=24000.,used_mib=12000.)
    (p/'telemetry.jsonl').write_text(json.dumps({'sample':telemetry})+'\n')
    write_json(p/'condition.json',dict(config=c,case=case,corpus_id='test-only',split='profile'))
    write_json(p/'source_manifest.json',runtime_source_manifest())
    write_json(p/'result.json',dict(status='complete',mode='gpu',drained=True,warmup=warm,measurement=measure,
        requests_finalized=len(rows),metrics=metrics([r for r in rows if r['phase']=='measure'],measure,c['deadline_ms']),
        worker_environment={'fixture':'not real GPU'},worker_after=dict(backend_attestation={'model_artifacts_verified':True},
        gpu_properties={'uuid':'GPU-fake','name':'Test GPU','memory_mib':24000.})))
    seal(p);return p


def test_formal_candidate_aggregates_repetitions_without_fabricated_capacity(tmp_path):
    from rosa_repro.contracts import PROFILE,validate
    paths=[formal_fixture(tmp_path,i) for i in range(3)]
    result=profile_candidate(paths);p=result['profile'];validate(p,PROFILE)
    assert result['status']=='candidate_requires_review' and not result['automatic_scheduler_import_allowed']
    assert p['memory_mib']==24000 and 'capacity_bound_hz' not in p
    assert len(p['points'])==1 and p['points'][0]['rate_hz']==10 and p['points'][0]['sample_count']==3600
    assert p['points'][0]['p99_ms']==1
    blocked=profile_candidate(paths[:2]);assert blocked['profile'] is None
    assert 'three_repetitions_per_rate_required' in blocked['blocking_reasons']


def test_reaudit_refuses_missing_scheduled_arrival_even_with_resealed_summary(tmp_path):
    from rosa_repro.profile_runner import seal
    p=formal_fixture(tmp_path,0);rows=[json.loads(s) for s in (p/'requests.jsonl').read_text().splitlines()]
    rows.pop();(p/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    r=json.loads((p/'result.json').read_text());r['requests_finalized']-=1;r['measurement']['successful_before_cutoff']-=1
    r['metrics']=metrics([x for x in rows if x['phase']=='measure'],r['measurement'],2000)
    (p/'result.json').write_text(json.dumps(r));(p/'evidence.json').unlink();seal(p)
    with pytest.raises(ValueError,match='arrivals missing'):reaudit(p)


def test_response_latency_stops_before_saving_action_artifacts(tmp_path,monkeypatch):
    import numpy as np,time
    original=np.savez_compressed;starts={}
    def save(path,**arrays):
        if Path(path).name.startswith('action-'):
            starts[Path(path).name]=time.monotonic_ns();time.sleep(.003)
        return original(path,**arrays)
    monkeypatch.setattr(np,'savez_compressed',save)
    r,out=run(tmp_path,c=config('s1'),stage='smoke')
    rows=[json.loads(s) for s in (out/'requests.jsonl').read_text().splitlines()]
    assert starts and all(x['finished_ns']<=starts[x['output_file']] for x in rows if x['status']=='ok')


def test_explicit_plan_context_are_bound_into_input_hash(tmp_path):
    from rosa_repro.component_runtime import s1_observation
    c=config('s1');c['payload_overlay']={'plan':'pick up the cup','context':{'execution_history':'saved replay context'}}
    r,out=run(tmp_path,c=c,stage='smoke')
    assert r['status']=='complete'
    rows=[json.loads(s) for s in (out/'requests.jsonl').read_text().splitlines()]
    expected={digest_value(dict(observation=observation(i+1),**c['payload_overlay'])) for i in range(4)}
    assert all(row['input_sha256'] in expected for row in rows)
    assert s1_observation(dict(observation=observation(1),**c['payload_overlay']))['task']==['CPU fixture\nPlan: pick up the cup']
