"""CPU-only R06 admission rejection and actual large-batch subprocess evidence."""
import argparse,asyncio,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.r06_protocol import TASKS,prepare
from rosa_repro.r06_runtime import run_cell,reaudit
from rosa_repro.r06_adoption import quality
from rosa_repro.profile_adoption import Evidence
from rosa_repro.paired import save_state_bank
from rosa_repro.regression import write_json
from rosa_repro.execution_plan import runtime_source_manifest


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=runtime_source_manifest();rows=[]
 async def run():
  for batch in (8,16):
   base=out/str(batch)
   for task,name in TASKS.items():save_state_bank(base/'banks'/task,[np.array([float(i+1),.2]) for i in range(batch+1)],task=name,source={'kind':'synthetic'})
   s=prepare(base/'banks',mode='cpu_dummy',phase='formal',selected_batch=batch,call_timeout_s=30,trial_timeout_s=180)
   write_json(base/'suite.json',s);mapping={c['id']:'trials/'+c['id'] for c in s['cells']};write_json(base/'directories.json',mapping)
   cell=f'T01-{batch}-selected'
   r=await run_cell(s,base/'banks',cell,base/mapping[cell],sim_python=sys.executable)
   assert r['status']=='complete' and r['batch_sizes']==[1,batch]
   assert len(s['cells'])==36
   rows.append(dict(batch=batch,directory=str((base/mapping[cell]).relative_to(out)),audit=reaudit(base/mapping[cell])))
   try:quality(Evidence(base),{'r06':dict(suite='suite.json',bank_root='banks',directories='directories.json')},{'batch_size':batch},None)
   except ValueError as exc:
    assert 'formal GPU' in str(exc);write_json(base/'adoption-rejection.json',dict(status='blocked',reason=str(exc),expected=True))
   else:raise AssertionError('CPU evidence admitted')
   print(json.dumps(dict(batch=batch,status='passed',actual_batches=r['batch_sizes'])),flush=True)
 asyncio.run(run());assert runtime_source_manifest()==source
 write_json(out/'verification.json',dict(status='passed',source_manifest=source,actual_CPU_R06_trials=2,CPU_adoptions_refused=2,
     actual_CPU_simulator_process_starts=26,rows=rows,GPU_execution=False,pretrained_weights_loaded=False,pod_access_attempted=False,real_LIBERO_tasks_run=0))
if __name__=='__main__':main()
