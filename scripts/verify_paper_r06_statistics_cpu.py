"""CPU-only release proof: condition coverage, real simulator subprocesses, raw reaudits and uncertainty."""
import argparse,asyncio,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.paper_conditions import generate,validate as validate_paper
from rosa_repro.r06_protocol import TASKS,prepare
from rosa_repro.r06_runtime import run_cell,reaudit,CPUTransport
from rosa_repro.paired import save_state_bank
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.regression import write_json,sha256
from rosa_repro.statistical_evaluation import paired_success,r06_report,latency_blocks


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=runtime_source_manifest()
 for n in (1,2,8):
  matrix=generate(gpu_count=n);assert validate_paper(matrix)==matrix;write_json(out/f'paper-{n}gpu.json',matrix)
 write_json(out/'paper-counts.json',{str(n):len(generate(gpu_count=n)['cells']) for n in (1,2,8)})
 for task,name in TASKS.items():
  save_state_bank(out/'banks'/task,[np.array([float(i+1),.2]) for i in range(5)],task=name,source=dict(kind='synthetic'),seed=42)
 suite=prepare(out/'banks',mode='cpu_dummy',phase='formal',selected_batch=4,trial_timeout_s=120,call_timeout_s=15)
 write_json(out/'suite.json',suite);directories={c['id']:'trials/'+c['id'] for c in suite['cells']};write_json(out/'directories.json',directories)
 results=[]
 async def execute():
  for cid in suite['randomized_order']:
   r=await run_cell(suite,out/'banks',cid,out/directories[cid],sim_python=sys.executable)
   assert r['status']=='complete' and r['successful_cases']==5,r
   audit=reaudit(out/directories[cid]);results.append(audit)
   print(json.dumps(dict(cell=cid,status=r['status'],batches=r['batch_sizes'])),flush=True)
  class Wrong(CPUTransport):
   def predict(self,*args):
    r=super().predict(*args);r[-1]['metadata']['robot_id']='wrong-robot';return r
  failed=await run_cell(suite,out/'banks','T01-4-selected',out/'intentional-wrong-response',sim_python=sys.executable,transport_factory=Wrong)
  assert failed['status']=='failed' and failed['planned_cases']==5 and failed['completed_cases']==0
  assert not list((out/'intentional-wrong-response').glob('cases/*/action*'))
 asyncio.run(execute())
 report=r06_report(suite,out,directories);assert report['status']=='complete' and len(report['comparisons'])==18
 assert all(c['statistics']['all_pairs_valid'] and c['statistics']['decision']=='unresolved' for c in report['comparisons'])
 write_json(out/'r06-report.json',report)
 failed_dirs=dict(directories);failed_dirs['T01-4-selected']='intentional-wrong-response';failed_dirs['T03-4-selected']='not-run'
 write_json(out/'incomplete-directories.json',failed_dirs);incomplete=r06_report(suite,out,failed_dirs)
 assert incomplete['status']=='incomplete' and {x['status'] for x in incomplete['rows']}=={'complete','failed','missing'}
 write_json(out/'incomplete-report.json',incomplete)
 # Actual CLI supervisor on one extra trial, and dry-run proving no process/model execution.
 base=[sys.executable,str(ROOT/'scripts/r06_experiments.py'),'run','--suite',str(out/'suite.json'),'--bank-root',str(out/'banks'),
       '--cell','T01-4-selected','--sim-python',sys.executable]
 subprocess.run(base+['--output',str(out/'dry-run.json')],check=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 subprocess.run(base+['--execute','--output',str(out/'supervised')],check=True,timeout=210,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 supervised=json.loads((out/'supervised/supervision.json').read_text());assert supervised['status']=='complete'
 # Independent analytic boundary: 50 concordant successes do not establish +/-5pp equivalence.
 fixed=[dict(task='synthetic',state_sha256=str(i),seed=i,status='completed',success=True,initial_observation_sha256=str(i),restored_state_sha256=str(i)) for i in range(50)]
 paired=paired_success(fixed,fixed);assert paired['confidence_interval'][0]<-.05 and paired['confidence_interval'][1]>.05
 write_json(out/'all-50-success-uncertainty.json',dict(provenance='synthetic_calculation_not_LIBERO',statistics=paired))
 raw=[dict(request_id=f'{block}-{robot}',cohort_ns=block*1_000_000_000+robot,status='ok',latency_ms=float(10+block%11)) for block in range(32) for robot in range(32)]
 raw.append(dict(request_id='timeout',cohort_ns=31_500_000_000,status='timeout',latency_ms=None))
 controls=dict(start_ns=0,end_ns=32_050_000_000,block_ns=1_000_000_000,resamples=500,seed=42)
 write_json(out/'synthetic-latency-input.json',dict(provenance='synthetic_not_GPU',controls=controls,rows=raw))
 latency=latency_blocks(raw,**controls);assert latency['status']=='estimated' and latency==latency_blocks(raw,**controls)
 assert latency['planned_logged_arrivals']==1025 and latency['status_counts']['timeout']==1
 write_json(out/'synthetic-latency-report.json',latency)
 paths=[out/d for d in directories.values()]+[out/'intentional-wrong-response',out/'supervised/raw']
 starts=0;allchecks=[]
 for path in paths:
  allchecks.append(reaudit(path));events=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
  starts+=sum(e['kind']=='simulator_start' for e in events)
  assert json.loads((path/'source_manifest.json').read_text())==source
 assert starts==144 and runtime_source_manifest()==source
 write_json(out/'verification.json',dict(status='passed',source_manifest=source,actual_CPU_trials=29,successful_matrix_trials=27,
     intentionally_failed_trials=1,supervised_success_trials=1,actual_CPU_simulator_process_starts=starts,
     fixed_cases_per_task=5,tasks=list(TASKS),arms=['official','shared_b1','selected'],concurrency=[1,2,4],
     matrix_successful_episodes=135,matrix_revalidated_actions=sum(r['actions_revalidated'] for r in results),
     paired_comparisons=18,all_checks=allchecks,raw_evidence_index_hashes={str(p.relative_to(out)):sha256(p/'evidence.json') for p in paths},
     GPU_execution=False,pretrained_weights_loaded=False,pod_access_attempted=False,Ray_execution=False,
     real_LIBERO_tasks_run=0,source_unchanged=True))
 print(json.dumps(dict(status='passed',trials=29,simulator_process_starts=starts)),flush=True)
if __name__=='__main__':main()
