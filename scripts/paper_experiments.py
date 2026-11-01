"""Generate or validate prospective R05/R08/R09 conditions; no network or execution."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.paper_conditions import generate,validate
from rosa_repro.regression import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',type=Path)
    p.add_argument('--gpu-count',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--sizes',type=Path)
    p.add_argument('--r05-selection',type=Path)
    p.add_argument('--dedicated-budgets',type=Path,help='JSON mapping dedicated method to its whole-workload GPU budget')
    a=p.parse_args();read=lambda f:json.loads(f.read_text()) if f else None
    result=validate(read(a.verify)) if a.verify else generate(seed=a.seed,repetitions=a.repetitions,
        gpu_count=a.gpu_count,sizes=read(a.sizes),r05_selection=read(a.r05_selection),dedicated_budgets=read(a.dedicated_budgets))
    write_json(a.output,result)
    print(json.dumps(dict(cells=len(result['cells']),sha256=result['sha256'],executable=False)))
if __name__=='__main__':main()
