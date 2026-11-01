"""Reaudit paired R06 outcomes or estimate latency uncertainty from time blocks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.statistical_evaluation import r06_report,latency_report
from rosa_repro.regression import write_json

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('r06');q.add_argument('--suite',type=Path,required=True);q.add_argument('--directories',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--margin',type=float,default=.05)
 q=sub.add_parser('latency');q.add_argument('--raw',type=Path,required=True);q.add_argument('--kind',choices=['profile','pipeline','r05'],required=True);q.add_argument('--block-ns',type=int,required=True);q.add_argument('--seed',type=int,default=42);q.add_argument('--resamples',type=int,default=2000)
 for q in sub.choices.values():q.add_argument('--confidence',type=float,default=.95);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args();read=lambda f:json.loads(f.read_text())
 if a.command=='r06':r=r06_report(read(a.suite),a.root,read(a.directories),confidence=a.confidence,margin=a.margin)
 else:r=latency_report(a.raw,kind=a.kind,block_ns=a.block_ns,seed=a.seed,resamples=a.resamples,confidence=a.confidence)
 write_json(a.output,r)
if __name__=='__main__':main()
