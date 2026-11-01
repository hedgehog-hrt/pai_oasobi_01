"""Real CPU subprocess/Ray/Torch workflow and P1-P4 reports. Never uses SSH/GPU."""
import argparse,json,sys,asyncio
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.profile_protocol import build_corpus,make_suite
from rosa_repro.regression import save_observation,write_json,sha256
from rosa_repro.dummy import observation
from rosa_repro.paired import digest_value
from rosa_repro.resume_workflow import prepare,run
from rosa_repro.profile_runner import profile_candidate,reaudit,verify_evidence
from rosa_repro.profile_adoption import inspect
from rosa_repro.comparison_report import condition,report,markdown
from verify_postpod_cpu import cpu_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--driver-python',required=True);p.add_argument('--release');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);images=out/'observations';c=cpu_config('s1')
    for i in range(4):save_observation(images/str(i),observation(i+1),dict(task='CPU workflow fixture',case={'state_sha256':digest_value([i])},execution_kind='cpu_dummy'))
    corpus=build_corpus(images,kind='synthetic_cpu');write_json(out/'config.json',c);suites={}
    for stage in ('smoke','pilot','profile'):
        path=out/(stage+'-suite.json');write_json(path,make_suite(c,corpus,stage=stage,phase='cpu_smoke',rates=[10] if stage=='profile' else []));suites[stage]=path
    plan=prepare(out/'config.json',images,suites,driver_python=str(Path(a.driver_python).absolute()),worker_python=sys.executable,release=a.release,setup_budget_s=120,kernel_budget_s=60)
    write_json(out/'workflow-plan.json',plan)
    dry=run(plan,out/'workflow');write_json(out/'dry-run.json',dry);assert dry['executed_steps']==0 and not (out/'workflow').exists()
    short=run(plan,out/'workflow',execute=True,budget_s=1);write_json(out/'budget-stop.json',short);assert short['status']=='budget_stop'
    first=run(plan,out/'workflow',execute=True,budget_s=3600,max_new=2);write_json(out/'first.json',first);assert first['status']=='step_limit'
    second=run(plan,out/'workflow',execute=True,budget_s=3600,max_new=4);write_json(out/'second.json',second);assert second['status']=='complete',second
    hashes={str(p.relative_to(out/'workflow')):sha256(p) for p in (out/'workflow').glob('*/attempt-*/artifact_hashes.json')}
    again=run(plan,out/'workflow',execute=True,budget_s=1);write_json(out/'resume-noop.json',again)
    assert again['status']=='complete' and again['executed_steps']==0
    assert hashes=={str(p.relative_to(out/'workflow')):sha256(p) for p in (out/'workflow').glob('*/attempt-*/artifact_hashes.json')}
    raw=list((out/'workflow').glob('*/attempt-*/probe/*/attempt-*/raw'))
    assert len(raw)==3
    for path in raw:assert reaudit(path)['successful']>0
    kernel=out/'workflow/kernel/attempt-001/kernel';verify_evidence(kernel);k=json.loads((kernel/'result.json').read_text())
    assert k['status']=='complete' and not k['gpu_execution'] and not k['gpu_kernel_events_present']
    profile=next(x for x in raw if json.loads((x/'condition.json').read_text())['case']['stage']=='profile')
    request=dict(schema_version='1.0',performance=[str(profile.relative_to(out))],kernel=str(kernel.relative_to(out)),quality={})
    try:inspect(request,out)
    except ValueError as e:write_json(out/'cpu-adoption-refused.json',dict(status='blocked',reason=str(e)))
    else:raise AssertionError('CPU evidence must not enter GPU R07')
    # CPU local-transport repetitions validate report wiring; separate real-Ray fleet verifier covers deployment.
    sys.path.insert(0,str(ROOT/'tests'))
    from runtime_helpers import plan_for
    from test_pipeline_runtime import run as run_pipeline
    fleet=plan_for(('P1','P2','P3','P4'))
    # Declare intended cells before executing, with conditions derived from a non-report calibration run.
    run_pipeline(fleet,out/'report-calibration');cond=condition(out/'report-calibration')
    cells=[dict(id='rep-'+str(i),condition_id='cpu-connection',variant='r07-cpu',repetition=i,directory='pipeline-'+str(i)) for i in range(3)]
    m=dict(schema_version='1.0',purpose='prospective_pipeline_comparison',conditions={'cpu-connection':cond},cells=cells)
    write_json(out/'comparison-manifest.json',m)
    for i in range(2):run_pipeline(fleet,out/('pipeline-'+str(i)))
    r=report(m,out);write_json(out/'comparison-report.json',r);(out/'comparison-report.md').write_text(markdown(r))
    assert [x['status'] for x in r['rows']]==['passed','passed','missing'] and not r['paper_reproduction_confirmed']
    v=dict(status='passed',source_manifest=runtime_source_manifest(),actual_CPU_Ray_trials=3,actual_CPU_torch_trace=True,
        dry_run_no_execution=True,budget_stop_before_execution=True,successful_steps_not_repeated=True,complete_resume_noop=True,
        raw_profile_reaudit_identical=True,CPU_profile_adoption_refused=True,expected_missing_repetition_preserved=True,
        report_pipeline_sets=['P1','P2','P3','P4'],immutable_installed_release_verified=bool(a.release),gpu_execution=False,pod_access_attempted=False,pretrained_weights_loaded=False)
    write_json(out/'verification.json',v);print(json.dumps({k:v for k,v in v.items() if k!='source_manifest'},indent=2))
if __name__=='__main__':main()
