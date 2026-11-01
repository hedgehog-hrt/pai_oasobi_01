"""Actual local Ray/Serve R05 proof. CPU fixtures only, no SSH or Pod operations."""
import argparse,json,subprocess,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.r05_protocol import template,timing,prepare
from rosa_repro.r05_runtime import reaudit
from rosa_repro.r05_workflow import check_pilots,report
from rosa_repro.paper_conditions import generate
from rosa_repro.profile_protocol import build_corpus
from rosa_repro.regression import write_json,save_observation,sha256
from rosa_repro.execution_plan import DUMMY_FACTORY,runtime_source_manifest,digest
from rosa_repro.dummy import observation
from rosa_repro.paired import digest_value
from rosa_repro.statistical_evaluation import latency_report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--single-development',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=runtime_source_manifest()
    corpus_root=out/'observations'
    for i in range(4):save_observation(corpus_root/str(i),observation(i+1),dict(task='synthetic-R05',case={'state_sha256':digest_value([i])},execution_kind='cpu_dummy'))
    book=build_corpus(corpus_root,kind='synthetic_cpu');write_json(out/'corpus.json',book)
    c=template();c.update(mode='cpu_dummy',hardware_kind='cpu-fixture',model_revision='synthetic',runtime_fingerprint='r05-CPU-v1',
        binding=dict(factory=DUMMY_FACTORY,kwargs=dict(delay_s=.24)),batch_wait_ms=30,watchdog_s=10,drain_s=10,setup_budget_s=90)
    write_json(out/'config.json',c);paper=generate();write_json(out/'paper.json',paper)
    protocol=timing('cpu_smoke',warmup_s=1.2,duration_s=2.2)
    pilot=prepare(paper,c,book,stage='pilot',groups=['k4-a50'],protocol=protocol);write_json(out/'pilot-suite.json',pilot)
    attempts=[]
    def command(name,suite_path,store,*extra,expected=(0,),pilots=None):
        cmd=[sys.executable,str(ROOT/'scripts/r05_experiments.py'),'run','--suite',str(suite_path),'--corpus-root',str(corpus_root),
             '--output',str(out/store),'--report',str(out/(name+'.json')),'--evidence-root',str(out),*extra]
        if pilots:cmd+=['--pilots',str(pilots)]
        with (out/(name+'.log')).open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=900)
        write_json(out/(name+'-process.json'),dict(argv=cmd,exit_code=result.returncode))
        assert result.returncode in expected,(name,result.returncode)
        value=json.loads((out/(name+'.json')).read_text());print(json.dumps(dict(step=name,status=value['status'],trials=value.get('executed_trials'))),flush=True)
        return value
    assert command('dry',out/'pilot-suite.json','pilot-runs')['status']=='planned' and not (out/'pilot-runs').exists()
    assert command('budget',out/'pilot-suite.json','pilot-runs','--execute','--budget-seconds','1',expected=(2,))['status']=='budget_stop'
    chosen=next(x for x in pilot['cells'] if x['batch_size']==4)
    first=command('first',out/'pilot-suite.json','pilot-runs','--execute','--budget-seconds','300','--cell',chosen['id'])
    assert first['status']=='complete' and first['executed_trials']==1
    raw=Path(first['rows'][0]['directory'])/'trial/raw';check=reaudit(raw);attempts.append(raw)
    assert '4' in check['metrics']['measurement']['actual_batch_histogram_requests']
    assert check['metrics']['measurement']['latency_ms_successful_response_cohort']['p50']>200
    assert check['metrics']['measurement']['completed_virtual_action_chunks']>0
    if a.single_development:
        write_json(out/'verification.json',dict(status='passed',source_manifest=source,actual_CPU_Ray_trials=1,GPU_execution=False,pod_access_attempted=False));return
    before=sha256(raw/'evidence.json')
    resumed=command('resume-one',out/'pilot-suite.json','pilot-runs','--execute','--budget-seconds','1','--cell',chosen['id'])
    assert resumed['executed_trials']==0 and sha256(raw/'evidence.json')==before
    rest=command('all-pilots',out/'pilot-suite.json','pilot-runs','--execute','--max-new','5','--budget-seconds','900')
    assert rest['status']=='complete' and rest['executed_trials']==4
    paths={cell['id']:f"pilot-runs/{cell['id']}/attempt-001/trial/raw" for cell in pilot['cells']}
    attempts=[out/p for p in paths.values()]
    pilot_report=report(pilot,out,paths);assert pilot_report['status']=='complete';write_json(out/'pilot-report.json',pilot_report);write_json(out/'pilot-directories.json',paths)
    deps={'k4-a50':list(paths.values())};write_json(out/'pilot-dependencies.json',deps)
    from rosa_repro.r05_prediction import generate as predict,verify as verify_prediction
    prediction=predict(out,deps);verify_prediction(prediction,out)
    subprocess.run([sys.executable,str(ROOT/'scripts/r05_experiments.py'),'predict','--pilots',str(out/'pilot-dependencies.json'),
        '--root',str(out),'--output',str(out/'prediction.json'),'--selection-output',str(out/'selection.json')],check=True,timeout=60,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert json.loads((out/'prediction.json').read_text())==prediction
    h=sha256(out/'prediction.json');text=(out/'prediction.json').read_text()
    confirm_paper=generate(r05_selection=json.loads((out/'selection.json').read_text()))
    confirm=prepare(confirm_paper,c,book,stage='confirmation',groups=['k4-a50'],protocol=protocol,prediction_artifacts={h:text})
    write_json(out/'confirmation-paper.json',confirm_paper);write_json(out/'confirmation-suite.json',confirm)
    check_pilots(confirm,out,deps)
    result=command('confirmation',out/'confirmation-suite.json','confirmation-runs','--execute','--max-new','6','--budget-seconds','900',pilots=out/'pilot-dependencies.json')
    assert result['status']=='complete' and result['executed_trials']==len(confirm['cells'])
    confirm_paths={cell['id']:f"confirmation-runs/{cell['id']}/attempt-001/trial/raw" for cell in confirm['cells']}
    attempts.extend(out/p for p in confirm_paths.values());write_json(out/'confirmation-directories.json',confirm_paths)
    comparison=report(confirm,out,confirm_paths);assert comparison['status']=='complete' and comparison['groups'][0]['paired_speedup']['summary']['n']==3
    write_json(out/'confirmation-report.json',comparison)
    missing=dict(confirm_paths);missing[confirm['cells'][0]['id']]='not-run';write_json(out/'missing-directories.json',missing)
    incomplete=report(confirm,out,missing);assert incomplete['groups'][0]['paired_speedup'] is None;write_json(out/'missing-report.json',incomplete)
    # B=16 is a cap, not a requirement to pad a one-robot batch; long action time is really waited.
    tail=prepare(paper,c,book,stage='pilot',groups=['k1-a1000'],protocol=timing('cpu_smoke',warmup_s=1.4,duration_s=2.8))
    write_json(out/'tail-suite.json',tail);cid=next(x['id'] for x in tail['cells'] if x['batch_size']==16)
    x=command('tail',out/'tail-suite.json','tail-runs','--execute','--budget-seconds','300','--cell',cid)
    tail_raw=Path(x['rows'][0]['directory'])/'trial/raw';attempts.append(tail_raw);a_tail=reaudit(tail_raw)
    assert a_tail['status']=='complete' and list(a_tail['metrics']['measurement']['actual_batch_histogram_requests'])==['1']
    # Warmup is rejected before the long measurement window; no synthetic zero throughput.
    short=prepare(paper,c,book,stage='pilot',groups=['k4-a50'],protocol=timing('cpu_smoke',warmup_s=.05,duration_s=20,warmup_min_success=50))
    write_json(out/'warmup-stop-suite.json',short);cid=next(x['id'] for x in short['cells'] if x['batch_size']==4)
    x=command('warmup-stop',out/'warmup-stop-suite.json','warmup-stop-runs','--execute','--budget-seconds','300','--cell',cid,expected=(2,))
    stop_raw=Path(x['rows'][0]['directory'])/'trial/raw';attempts.append(stop_raw);stopped=reaudit(stop_raw)
    assert stopped['status']=='warmup_shortfall' and not stopped['measurement_started'] and stopped['metrics'] is None
    badc=dict(c,watchdog_s=.03,drain_s=2)
    bad=prepare(paper,badc,book,stage='pilot',groups=['k4-a50'],protocol=protocol);write_json(out/'watchdog-suite.json',bad)
    cid=next(x['id'] for x in bad['cells'] if x['batch_size']==4)
    x=command('watchdog',out/'watchdog-suite.json','watchdog-runs','--execute','--budget-seconds','300','--cell',cid,expected=(2,))
    failed_raw=Path(x['rows'][0]['directory'])/'trial/raw';attempts.append(failed_raw);failed=reaudit(failed_raw)
    assert failed['status']=='failed' and not failed['measurement_started'] and failed['drained']
    ev=[json.loads(x) for x in (failed_raw/'events.jsonl').read_text().splitlines()]
    assert any(x['kind']=='late_response' for x in ev) and not any(x['kind']=='action_start' for x in ev)
    latency=latency_report(raw,kind='r05',block_ns=250000000,resamples=200);write_json(out/'latency-block-report.json',latency)
    checks=[]
    for path in attempts:
        audit=reaudit(path);checks.append(dict(directory=str(path.relative_to(out)),audit=audit))
        assert json.loads((path/'source_manifest.json').read_text())==source
        assert json.loads((path.parent/'cleanup.json').read_text())['own_deployments_removed']
    assert len(attempts)==8+len(confirm['cells']) and runtime_source_manifest()==source
    write_json(out/'verification.json',dict(status='passed',source_manifest=source,actual_CPU_Ray_trials=len(attempts),complete_trials=len(attempts)-2,
        expected_warmup_shortfall_trials=1,expected_watchdog_failure_trials=1,checks=checks,
        prediction_generated_and_recomputed=True,selected_batch=prediction['groups']['k4-a50'],
        primary_metric='completed_virtual_action_chunks_per_second',valid_over_200ms_responses_used=True,
        confirmation_independent_repetitions=3,all_5_pilot_batches_required=True,batch16_single_robot_executed_as_batch1=True,
        exact_1s_action_wait_reaudited=True,no_fake_SLO=True,resume_skips_completed=True,source_unchanged=True,
        GPU_execution=False,pretrained_weights_loaded=False,pod_access_attempted=False,real_LIBERO_tasks_run=0))
    print(json.dumps(dict(status='passed',actual_CPU_Ray_trials=len(attempts))),flush=True)
if __name__=='__main__':main()
