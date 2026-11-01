"""Save exact reference/shared S1 actions from an idle pinned server, then re-audit."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.regression import load_observation,sha256,write_json
from rosa_repro.action_regression import save_case,reaudit_case,validate_run,seal_run,comparison_controls
from rosa_repro.execution_plan import runtime_source_manifest,digest
from rosa_repro.paired import digest_value


def capture(client,observations,output,*,seed=42,atol=0.,rtol=0.,batch_size=1,comparison='same_shape'):
    manifests=sorted(Path(observations).rglob('observation.json'))
    if not manifests or type(batch_size) is not int or not 1<=batch_size<=64 or len(manifests)%batch_size:
        raise ValueError('Nonempty observation count must be divisible by declared batch size; no discarded tail/padding')
    if type(seed) is not int or not 0<=seed<2**32-len(manifests):raise ValueError('Invalid seed range')
    import math
    if not all(math.isfinite(v) and v>=0 for v in (atol,rtol)):raise ValueError('Invalid tolerances')
    comparison_controls(batch_size,comparison)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);expected=len(manifests)//batch_size
    context=dict(comparison=comparison,seed=seed,atol=atol,rtol=rtol,batch_size=batch_size,expected_cases=expected,
        inputs=[dict(directory=str(p.parent.relative_to(observations)),manifest_sha256=sha256(p),observation_sha256=digest_value(load_observation(p.parent)[0])) for p in manifests],
        source_manifest=runtime_source_manifest(),latency_benchmark=False)
    write_json(out/'run_context.json',context);rows=[];status='failed';server={};error=None
    try:
        server=client.call_endpoint('r02_runtime',requires_input=False);write_json(out/'server_evidence.json',server)
        if server.get('rosa_source_manifest')!=runtime_source_manifest():raise ValueError('Updated array-capable server with matching source required')
        for index in range(expected):
            selected=manifests[index*batch_size:(index+1)*batch_size];loaded=[load_observation(p.parent) for p in selected]
            obs=[x[0] for x in loaded]
            response=client.call_endpoint('r02_compare_arrays',dict(observations=obs,seed=seed+index*batch_size,atol=atol,rtol=rtol,comparison=comparison))
            if response.get('comparison')!=comparison:raise ValueError('Server changed comparison purpose')
            if (response['seed'],response['atol'],response['rtol'])!=(seed+index*batch_size,atol,rtol):raise ValueError('Server changed requested numerical controls')
            directory=f'raw-case-{index:04d}';save_case(out/directory,obs,response,provenance=[x[1]['provenance'] for x in loaded],server_evidence=server)
            r=reaudit_case(out/directory)
            row=dict(passed=r['passed'],same_loaded_model=r['same_loaded_model'],default_rngs_restored=r['default_rngs_restored'],
                raw_case_directory=directory,raw_case_sha256=sha256(out/directory/'case.json'),
                comparison={k:v for k,v in r.items() if k not in ('source_manifest','server_evidence')})
            write_json(out/f'case-{index:04d}.json',row);rows.append(row)
        status='passed' if all(x['passed'] for x in rows) else 'mismatch'
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        write_json(out/'comparison_summary.json',dict(schema_version='3.0',comparison=comparison,numerical_admission_eligible=comparison=='same_shape' and status=='passed',
            status=status,cases=len(rows),expected_cases=expected,results=rows,error=error,
            gpu_execution_confirmed_by_response=bool(rows) and server.get('runtime',{}).get('observed_torch_state',{}).get('cuda_available') is True,
            failed_or_timed_out_request_execution_unknown=status=='failed',task_success_measured=False,latency_benchmark=False,
            raw_arrays_saved=True,stochastic_batch_equivalence_verified=False))
        seal_run(out)
    if status in ('passed','mismatch'):validate_run(out,expected_batch_size=batch_size,require_passed=False,required_comparison=comparison)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path);p.add_argument('--observations',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=5556);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--atol',type=float,default=0.);p.add_argument('--rtol',type=float,default=0.);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--comparison',choices=['same_shape','serial_vs_batch'],default='same_shape')
    p.add_argument('--reaudit',type=Path);a=p.parse_args()
    if a.reaudit:
        result=validate_run(a.reaudit,current_source=False,expected_batch_size=a.batch_size,require_passed=False)
        write_json(a.output,result);return 0 if result['status']=='passed' else 2
    if not a.repo or not a.observations:p.error('--repo and --observations required')
    sys.path.insert(0,str(a.repo.resolve()))
    from gr00t.policy.server_client import PolicyClient
    with PolicyClient(host='127.0.0.1',port=a.port,timeout_ms=120000) as client:
        status=capture(client,a.observations,a.output,seed=a.seed,atol=a.atol,rtol=a.rtol,batch_size=a.batch_size,comparison=a.comparison)
    return 0 if status=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
