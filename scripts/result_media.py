"""Offline audited figures and independent explanation-run video, without model downloads."""
import argparse,asyncio,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.regression import write_json

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
 q=s.add_parser('figures');q.add_argument('--request',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 q=s.add_parser('video');q.add_argument('--capture',type=Path,required=True);q.add_argument('--camera',required=True);q.add_argument('--fps',type=float,default=5);q.add_argument('--output',type=Path,required=True)
 q=s.add_parser('record-cpu');q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='figures':
  from rosa_repro.result_figures import export
  r=export(json.loads(a.request.read_text()),a.root,a.output)
 elif a.cmd=='video':
  from rosa_repro.explanation_video import export
  r=export(a.capture,a.output,camera=a.camera,fps=a.fps)
 else:
  from rosa_repro.simulator_process import ChangingDummyEpisode
  from rosa_repro.explanation_video import record
  from rosa_repro.paired import digest_value
  import numpy as np
  class Sim:
   def __init__(self):self.env=ChangingDummyEpisode()
   async def reset(self,state,seed):
    obs,h=self.env.reset(state,seed);return dict(observation=obs,restored_state_sha256=h)
   async def step(self,a,n):return self.env.step(a,n)
  async def predict(obs):return {'action.x':np.full((1,8,1),obs['state.x'][0,0,0],dtype=np.float32)}
  r=asyncio.run(record(Sim(),predict,a.output,state=np.array([1.]),seed=42,mode='cpu_dummy',provenance={'purpose':'CPU recording fixture'}))
 print(json.dumps({k:v for k,v in r.items() if k in ('status','figures','input_status','video','frames','mode','episode_done','performance_eligible')},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
