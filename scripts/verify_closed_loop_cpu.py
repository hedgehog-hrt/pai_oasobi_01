"""Local CPU Ray workers + separate simulator processes; no GPU/SSH/Pod operations."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.closed_loop_protocol import protocol,prepare_suite
from rosa_repro.closed_loop_runtime import reaudit_closed
from rosa_repro.execution_plan import runtime_source_manifest,digest
from rosa_repro.paired import save_state_bank
from rosa_repro.regression import write_json,sha256,save_observation
from rosa_repro.comparison_report import report,markdown
from rosa_repro.action_regression import collect_arrays,validate_run
from rosa_repro.dummy import DummyPolicy,observation
from r02_compare import capture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--driver-python',required=True);p.add_argument('--sim-python',required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'examples/runtime_mix4_plan.json').read_text())
    if plan['mode']!='cpu_dummy':raise ValueError('Verifier is CPU only')
    save_state_bank(out/'bank',[np.array([1.,.1]),np.array([4.,.5]),np.array([8.,.9])],task='synthetic live CPU',source={'kind':'synthetic'})
    pro=protocol(plan,'cpu_smoke',warmup_s=3.2,duration_s=6.2,request_timeout_s=10,sim_timeout_s=10,drain_timeout_s=5,warmup_min_success=1)
    suite=prepare_suite(plan,{c['audit']['id']:'bank' for c in plan['classes']},out,pro,repetitions=2,setup_budget_s=120)
    write_json(out/'suite.json',suite)
    condensed=dict(pro,source_provenance={cid:dict(manifest_sha256=e['manifest_sha256'],states_sha256=e['states_sha256']) for cid,e in suite['banks'].items()})
    cond=dict(execution_mode=plan['mode'],action_semantics=pro['action_semantics'],source_sha256=runtime_source_manifest()['sha256'],
        workload_sha256=digest(dict(classes=plan['inputs']['fleet']['classes'],hardware=plan['inputs']['fleet']['hardware'])),protocol_sha256=digest(condensed))
    manifest=dict(schema_version='1.0',purpose='prospective_pipeline_comparison',conditions={'live-cpu':cond},
        cells=[dict(id=f'rep-{i}',condition_id='live-cpu',variant='r07-cpu-live',repetition=i,
            directory=f'runs/closed-loop-{i:03d}/attempt-001/trial/raw') for i in range(3)])
    write_json(out/'comparison-manifest.json',manifest)
    def command(name,*flags,suite_path=None,run_name='runs'):
        argv=[str(Path(a.driver_python).absolute()),str(ROOT/'scripts/pipeline_experiments.py'),'run','--suite',str(suite_path or out/'suite.json'),
            '--root',str(out),'--output',str(out/run_name),'--sim-python',str(Path(a.sim_python).absolute()),'--report',str(out/(name+'.json')),*flags]
        with (out/(name+'.log')).open('x') as log:
            result=subprocess.run(argv,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,timeout=420)
        write_json(out/(name+'-process.json'),dict(argv=argv,exit_code=result.returncode))
        return json.loads((out/(name+'.json')).read_text())
    assert command('dry')['status']=='planned' and not (out/'runs').exists()
    assert command('budget','--execute','--budget-seconds','1')['status']=='budget_stop'
    first=command('first','--execute','--budget-seconds','400');assert first['status']=='trial_limit',first
    second=command('second','--execute','--budget-seconds','400');assert second['status']=='complete',second
    raw_paths=sorted((out/'runs').glob('*/attempt-*/trial/raw'));assert len(raw_paths)==2
    hashes={str(p):sha256(p/'evidence.json') for p in raw_paths}
    again=command('resume','--execute','--budget-seconds','1');assert again['status']=='complete' and again['executed_trials']==0
    assert hashes=={str(p):sha256(p/'evidence.json') for p in raw_paths}
    summaries=[]
    for raw in raw_paths:
        r=reaudit_closed(raw);workers=json.loads((raw/'workers_after.json').read_text());sims=json.loads((raw/'simulators.json').read_text())
        assert len({(w['node_id'],w['pid']) for w in workers.values()})==4 and len({v['pid'] for v in sims.values()})==4
        assert all(v['kind']=='cpu_dummy' for v in sims.values()) and r['actions_revalidated']>0
        assert r['measurement_started'] and r['warmup_gate']['requirements_met']
        details=[json.loads(x) for x in (raw/'request_details.jsonl').read_text().splitlines()]
        assert any(x['observation_version']>0 for x in details)
        assert all(v['all_started']>1 for v in r['tasks']['classes'].values())
        summaries.append(dict(actions=r['actions_revalidated'],tasks=r['tasks'],samples=r['samples']))
    report_value=report(manifest,out);assert [x['status'] for x in report_value['rows']]==['passed','passed','missing'],report_value
    write_json(out/'comparison-report.json',report_value);(out/'comparison-report.md').write_text(markdown(report_value))
    rejected=protocol(plan,'cpu_smoke',warmup_s=.2,duration_s=20,request_timeout_s=10,sim_timeout_s=10,drain_timeout_s=5,warmup_min_success=1)
    stop_suite=prepare_suite(plan,{c['audit']['id']:'bank' for c in plan['classes']},out,rejected,repetitions=1,setup_budget_s=120)
    write_json(out/'warmup-stop-suite.json',stop_suite)
    stop_result=command('warmup-stop','--execute','--budget-seconds','400',suite_path=out/'warmup-stop-suite.json',run_name='warmup-stop-runs')
    assert stop_result['status']=='failed' and stop_result['executed_trials']==1
    stop_raw=out/'warmup-stop-runs/closed-loop-000/attempt-001/trial/raw'
    stop_audit=reaudit_closed(stop_raw);stop= json.loads((stop_raw/'result.json').read_text());audit=json.loads((stop_raw/'audit.json').read_text())
    assert stop['status']=='warmup_shortfall' and not stop['measurement_started'] and stop['drained'] and stop['weighted_Q'] is None
    assert stop['finished_ns']<audit['window_end_ns'] and not stop_audit['warmup_gate']['requirements_met']
    assert all((x['sent_ns'] if x['record_type']=='request' else x['start_ns'])<audit['window_start_ns'] for x in audit['records'])
    assert json.loads((stop_raw.parent/'cleanup.json').read_text())['status']=='complete'
    write_json(out/'warmup-stop-reaudit.json',stop_audit)
    class StochasticCPU(DummyPolicy):
        def get_action(self,obs):
            import torch,random
            action,_=super().get_action(obs)
            action['action.x']+=torch.rand(action['action.x'].shape).numpy()+np.random.random()+random.random()
            return action,{}
    class CPUClient:
        def call_endpoint(self,name,params=None,**kwargs):
            if name=='r02_runtime':return dict(rosa_source_manifest=runtime_source_manifest(),kind='synthetic_cpu_inprocess',
                runtime={'observed_torch_state':{'cuda_available':False}})
            return collect_arrays(StochasticCPU(),**params)
    for i in range(4):save_observation(out/'array-inputs'/str(i),observation(i+1),{'kind':'synthetic_cpu'})
    for batch in (1,2,4):
        target=out/f'actions-b{batch}'
        assert capture(CPUClient(),out/'array-inputs',target,batch_size=batch)=='passed'
        write_json(out/f'actions-b{batch}-reaudit.json',validate_run(target,expected_batch_size=batch))
        if batch>1:
            diagnostic=out/f'diagnostic-b{batch}'
            assert capture(CPUClient(),out/'array-inputs',diagnostic,batch_size=batch,comparison='serial_vs_batch')=='mismatch'
            write_json(out/f'diagnostic-b{batch}-reaudit.json',validate_run(diagnostic,expected_batch_size=batch,require_passed=False))
            try:validate_run(diagnostic,expected_batch_size=batch)
            except ValueError as exc:write_json(out/f'diagnostic-b{batch}-admission-refused.json',dict(reason=str(exc)))
            else:raise AssertionError('Diagnostic used as numerical admission')
    result=dict(status='passed',source_manifest=runtime_source_manifest(),actual_CPU_Ray_trials=3,completed_measurement_trials=2,expected_warmup_stop_trials=1,distinct_workers_per_trial=4,
        distinct_simulator_processes_per_trial=4,live_observation_update=True,episode_reset=True,raw_log_reaudit=True,
        budget_stop_before_launch=True,resume_without_repeating=True,missing_comparison_repetition_retained=True,
        numerical_action_batches=[1,2,4],numerical_actions_recomputed=True,numerical_transport='CPU inprocess fixture, not GR00T/ZMQ',
        same_shape_stochastic_regression_passed=True,cross_shape_mismatches_preserved=True,cross_shape_not_admitted_as_numerical_regression=True,
        warmup_rejected_before_measurement=True,warmup_rejection_drained_and_cleaned=True,no_new_requests_or_actions_after_rejected_warmup=True,
        closed_loop_summaries=summaries,GPU_execution=False,pretrained_weights_loaded=False,pod_access_attempted=False)
    write_json(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_manifest','closed_loop_summaries')},indent=2))
if __name__=='__main__':main()
