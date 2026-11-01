"""Prepare auditable dedicated-GPU allocations; does not start models."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.dedicated_baselines import prepare
from rosa_repro.dedicated_execution import compile_plan
from rosa_repro.regression import write_json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--profiles',type=Path,required=True)
p.add_argument('--method',choices=['gpu_per_robot','gpu_per_model_per_robot'],required=True);p.add_argument('--gpu-budget',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--compile',action='store_true',help='Compile physical-group execution plan; starts no models')
if __name__=='__main__':
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 value=prepare(json.loads(a.reference.read_text()),a.method,json.loads(a.profiles.read_text()),gpu_budget=a.gpu_budget)
 write_json(a.output,compile_plan(value) if a.compile else value)
