"""Real CPU Ray + synthetic observation checks for the three pre-Pod preparations."""
import argparse,asyncio,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from observations import execute_quality
from verify_postpod_cpu import cpu_config
from rosa_repro.capture_plan import default_design,make_capture_plan,capture
from rosa_repro.checkpoint_verification import groot_pin
from rosa_repro.dummy import DummyEpisode,DummyTransport
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.observation_quality import prepare_quality,annotation_template,evaluate_quality
from rosa_repro.paired import save_state_bank
from rosa_repro.profile_protocol import build_corpus,template
from rosa_repro.profile_runner import verify_evidence
from rosa_repro.profiling_budget import formal_timing
from rosa_repro.regression import write_json


async def verify(out):
    bank=out/'state-bank';save_state_bank(bank,[np.array([i+1.,.3]) for i in range(30)],task='CPU fixture',source={'kind':'synthetic'})
    plan=make_capture_plan([bank],dict(default_design(),max_steps=9))
    capture_root=out/'capture';captured=capture(plan,[bank],capture_root,lambda t:DummyEpisode(),
        lambda:DummyTransport('official'),execution_kind='cpu_dummy')
    assert captured['completed_episodes']==30 and captured['observations']==60
    assert all(o['termination']=='step_limit' and not o['simulator_success'] for o in captured['outcomes'])
    corpus=build_corpus(capture_root,kind='synthetic_cpu');write_json(out/'corpus.json',corpus);reports=[]
    for role in ('s2','safety','monitor'):
        suite=prepare_quality(cpu_config(role),corpus,capture_root);write_json(out/(role+'-quality-suite.json'),suite)
        write_json(out/(role+'-annotations-blank.json'),annotation_template(suite))
        predictions=await execute_quality(suite,capture_root,out/(role+'-quality'))
        verify_evidence(out/(role+'-quality'))
        assert predictions['status']=='complete' and predictions['drained'] and len(predictions['rows'])==18
        assert all(x['status']=='ok' for x in predictions['rows'])
        quality=evaluate_quality(suite,predictions);assert quality['status']=='incomplete'
        assert quality['metrics']['unreviewed']==18
        write_json(out/(role+'-unreviewed-evaluation.json'),quality)
        reports.append(dict(role=role,requests=18,all_outputs_valid=True,semantic_status=quality['status']))
    timings={str(rate):formal_timing(rate,template('s2')) for rate in (.01,.05,.1,.5,1.,2.,10.)}
    assert not timings['0.01']['executable'] and timings['0.5']['protocol']['measure_max_s']==2446
    write_json(out/'low-rate-budgets.json',timings)
    pin=groot_pin();assert pin['weights_total_bytes']==6910499416
    import ray
    result=dict(status='passed',source_manifest=runtime_source_manifest(),actual_ray_version=ray.__version__,
        pod_access_attempted=False,gpu_execution=False,pretrained_weights_loaded=False,
        real_GR00T_weight_files_verified=False,GR00T_pin_indexed_weight_files=2,GR00T_pin_inference_files=7,
        bounded_synthetic_capture_episodes=30,synthetic_observation_frames=60,split_state_groups=dict(profile=15,evaluation=6,holdout=9),
        actual_CPU_Ray_semantic_requests=54,quality_runs=reports,blank_annotations_cannot_pass=True,
        simulator_step_limit_not_used_as_failed_label=True,low_rate_budget_precomputed=True,
        actual_low_rate_long_running_trial_performed=False,ray_driver_disconnected=not ray.is_initialized())
    write_json(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_manifest'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    asyncio.run(verify(a.output))
