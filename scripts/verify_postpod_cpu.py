"""Exercise the post-Pod CLI runner on real CPU Ray, including resume and audit."""
import argparse,asyncio,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from postpod_models import execute,check_smoke
from rosa_repro.profile_protocol import template,build_corpus,make_suite
from rosa_repro.profile_runner import reaudit,profile_candidate,pilot_rate_hints,compare_probes
from rosa_repro.regression import save_observation,write_json,sha256
from rosa_repro.paired import digest_value
from rosa_repro.dummy import observation
from rosa_repro.execution_plan import DUMMY_FACTORY,runtime_source_manifest


def cpu_config(role):
    c=template(role,'groot' if role=='s1' else 'vllm')
    c.update(mode='cpu_dummy',model_revision='synthetic-postpod-v1',runtime_fingerprint='cpu-test-v1',hardware_kind='cpu-fixture',
        streams=4,request_timeout_s=5.,drain_timeout_s=5.,max_dispatch_lag_s=.1,
        binding=dict(factory=DUMMY_FACTORY,kwargs=dict(delay_s=.02)))
    if role=='s1':c.update(batch_size=4,batch_wait_ms=30)
    if role=='monitor':c['binding']['kwargs']['async_concurrency']=2
    return c


async def verify(out):
    corpus_root=out/'observations'
    for state in range(4):
        for arm in ('official','shared'):
            save_observation(corpus_root/str(state)/arm,observation(state+1),dict(task='CPU fixture',case={'state_sha256':digest_value([state])},execution_kind='cpu_dummy'))
    corpus=build_corpus(corpus_root,kind='synthetic_cpu');write_json(out/'corpus.json',corpus)
    reports=[]
    # Two distinct repetitions verify that max-new and resume preserve attempts.
    s=make_suite(cpu_config('s1'),corpus,stage='smoke',phase='cpu_smoke',repetitions=2)
    write_json(out/'s1-suite.json',s);root=out/'s1-smoke'
    first=await execute(s,root,corpus_root,max_new=1)
    path0=Path(first['results'][0]['directory'])/'raw';original=sha256(path0/'evidence.json')
    second=await execute(s,root,corpus_root,max_new=1)
    third=await execute(s,root,corpus_root,max_new=1)
    assert len(first['results'])==len(second['results'])==1 and not third['results']
    assert sha256(path0/'evidence.json')==original
    paths=[path0,Path(second['results'][0]['directory'])/'raw']
    for path in paths:
        m=reaudit(path);assert m['successful']==m['arrivals'] and m['successful']>0
        assert '4' in m['batch_histogram_requests']
    comparison=compare_probes(*paths);assert all(x['exact_output_sets_equal'] for x in comparison['rows'])
    write_json(out/'s1-comparison.json',comparison)
    reports.extend(first['results']+second['results'])
    sm=make_suite(cpu_config('monitor'),corpus,stage='pilot',phase='cpu_smoke')
    write_json(out/'monitor-suite.json',sm)
    monitor=await execute(sm,out/'monitor-pilot',corpus_root,max_new=1)
    mp=Path(monitor['results'][0]['directory'])/'raw';mr=json.loads((mp/'result.json').read_text())
    assert mr['worker_after']['max_inflight_batches']==2
    hints=pilot_rate_hints(mp);assert hints['observed_completion_rate_hz']>0 and not hints['capacity_certified']
    write_json(out/'pilot-rate-hints.json',hints)
    reports.extend(monitor['results'])
    sp=make_suite(cpu_config('safety'),corpus,stage='profile',phase='cpu_smoke',rates=[20.])
    write_json(out/'safety-suite.json',sp)
    safety=await execute(sp,out/'safety-profile',corpus_root,max_new=1)
    safety_path=Path(safety['results'][0]['directory'])/'raw';assert reaudit(safety_path)['successful']>0
    candidate=profile_candidate([safety_path]);assert candidate['profile'] is None and not candidate['automatic_scheduler_import_allowed']
    write_json(out/'cpu-profile-rejected.json',candidate);reports.extend(safety['results'])
    assert all(r['status']=='complete' and r['all_arrivals_successful'] for r in reports)
    import ray
    result=dict(status='passed',source_manifest=runtime_source_manifest(),actual_ray_version=ray.__version__,
        gpu_execution=False,pod_access_attempted=False,pretrained_weights_loaded=False,
        stages_exercised=['smoke','pilot','profile'],s1_batch_four_observed=True,async_monitor_concurrency_observed=2,
        resumed_completed_attempt_unchanged=True,completed_suite_resume_is_noop=True,
        raw_reaudit_matches=True,cpu_measured_profile_export_rejected=True,
        exact_synthetic_S1_output_comparison=True,rate_hints_are_not_capacity_certificate=True,
        ray_driver_disconnected=not ray.is_initialized(),trials=reports)
    write_json(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_manifest','trials')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    asyncio.run(verify(a.output))
