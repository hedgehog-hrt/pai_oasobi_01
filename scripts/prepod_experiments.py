"""Generate GPU-ready condition definitions or execute a resumable CPU smoke suite."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import sys

os.environ['RAY_USAGE_STATS_ENABLED']='0'
os.environ['RAY_AUTH_TOKEN']=secrets.token_hex(32)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.experiments import (build_matrix,stable_id,ExperimentStore,summarize_store,
                                    split_observation_manifest)
from rosa_repro.regression import write_json,sha256


def source_fingerprint():
    root=Path(__file__).resolve().parents[1]
    return stable_id({str(p.relative_to(root)):sha256(p) for folder in ('src','scripts')
                     for p in sorted((root/folder).rglob('*.py'))})


async def execute(matrix,output,max_new=None):
    import ray
    from ray import serve
    from rosa_repro.ray_service import SharedModel
    from rosa_repro.replay import run_trial
    from rosa_repro.dummy import observation
    if any(c['execution_kind']!='cpu_dummy' or c['phase']!='cpu_smoke' for c in matrix['cases']):
        raise ValueError('This local executor only runs explicitly labelled CPU smoke trials')
    # Runtime and source must match before resuming; new code gets a new matrix.
    matrix=dict(matrix,driver_source_sha256=source_fingerprint(),ray_version=ray.__version__)
    store=ExperimentStore(output,matrix)
    completed_now=0
    started_ray=False
    try:
        pending=[c for c in matrix['cases'] if not store.completed(c['case_id'])]
        if not pending or max_new==0:return dict(new_trials=0,attempts=store.report())
        ray.init(num_cpus=4,include_dashboard=False,_node_ip_address='127.0.0.1',
                 _temp_dir='/private/tmp/rosa-trial-'+str(os.getpid()),log_to_driver=False)
        started_ray=True
        serve.start(proxy_location='Disabled')
        for case in pending:
            if max_new is not None and completed_now>=max_new:break
            # Fresh replica per condition prevents cross-condition queued work.
            name='rosa-'+case['case_id']
            handle=serve.run(SharedModel.bind(max_batch_size=case['batch_size'],
                batch_wait_s=case['batch_wait_s'],dummy_delay_s=.002),name=name,route_prefix=None)
            try:
                with store.attempt(case['case_id']) as directory:
                    result=await run_trial(handle,[observation(i+1) for i in range(10)],case,directory)
                    print(json.dumps(dict(case_id=case['case_id'],status=result['status'],
                                          metrics=result.get('metrics',{})),ensure_ascii=False),flush=True)
            finally:serve.delete(name)
            completed_now+=1
        return dict(new_trials=completed_now,attempts=store.report())
    finally:
        report=store.report()
        # Derived summary can be regenerated; raw attempts remain immutable.
        (Path(output)/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        store.close()
        if started_ray:
            serve.shutdown();ray.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['generate','cpu-smoke','resume','summarize','split-observations'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--matrix',type=Path)
    p.add_argument('--capacities',type=Path)
    p.add_argument('--observations',type=Path)
    p.add_argument('--max-new',type=int)
    a=p.parse_args()
    if a.command=='summarize':
        report=summarize_store(a.output)
        print(json.dumps({'cases':len(report['cases']),'statuses':[r['status'] for r in report['cases']]}))
        return
    if a.command=='split-observations':
        if a.observations is None:p.error('--observations required')
        write_json(a.output,split_observation_manifest(a.observations))
        return
    if a.max_new is not None and a.max_new<0:p.error('--max-new cannot be negative')
    if a.command=='generate':
        caps=json.loads(a.capacities.read_text()) if a.capacities else {}
        matrix=build_matrix(capacities=caps,observation_fingerprint='PENDING_REAL_OBSERVATIONS',
            runtime_fingerprint='PENDING_GPU_RUNTIME',execution_kind='gpu_replay',phase='formal',rate_cap_hz=None)
        matrix.update(executable=False,note='Resolve runtime/observation fingerprints and capacity evidence before GPU execution')
        write_json(a.output,matrix)
        print(json.dumps({'cases':len(matrix['cases']),'unresolved':matrix['unresolved'],'timing_budget_s':matrix['timing_budget_s']}))
        return
    if a.command=='cpu-smoke':
        # Values below are load-generator test inputs, NOT measured capacity.
        caps={'1':dict(hz=100.,evidence_sha256=stable_id({'kind':'synthetic_input','hz':100.}),execution_kind='cpu_dummy')}
        matrix=build_matrix(capacities=caps,observation_fingerprint=stable_id({'fixture':'dummy_observations_1_to_10'}),
            runtime_fingerprint='ray-2.59.0-cpu-dummy-delay-2ms',batches=(1,),robots=(1,4),repetitions=1,rate_cap_hz=2.)
        # Exercise low/high open-loop rates and all four closed-loop conditions.
        matrix['cases']=[c for c in matrix['cases'] if c['stage']=='R04' or c.get('load_fraction') in (.1,1.)]
        # Include an actual batch-four condition in the full executor path.
        from copy import deepcopy
        extra=deepcopy(next(c for c in matrix['cases'] if c['stage']=='R04' and c['robots']==4 and c['rate_cap_hz'] is None))
        extra.pop('case_id');extra['batch_size']=4;extra['case_id']='r04-'+stable_id(extra)[:16]
        matrix['cases'].append(extra)
        matrix['timing_budget_s']={'note':'CPU smoke subset; not a GPU-time estimate'}
    else:
        source=a.matrix or a.output/'matrix.json'
        matrix=json.loads(source.read_text())
        matrix.pop('driver_source_sha256',None);matrix.pop('ray_version',None)
    print(asyncio.run(execute(matrix,a.output,a.max_new)))
    summarize_store(a.output)


if __name__=='__main__':main()
