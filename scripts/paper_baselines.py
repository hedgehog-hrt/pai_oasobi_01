"""Bind R08/R09 cells to fixed allocation plans or re-audit complete comparisons."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.paper_baselines import compile_baseline
from rosa_repro.paper_comparison import report
from rosa_repro.regression import write_json
read=lambda p:json.loads(p.read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('plan');q.add_argument('--paper',type=Path,required=True);q.add_argument('--reference',type=Path,required=True);q.add_argument('--cell',required=True);q.add_argument('--profiles',type=Path);q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('report');q.add_argument('--paper',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--directories',type=Path,required=True);q.add_argument('--references',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args();r=compile_baseline(read(a.reference),read(a.paper),a.cell,read(a.profiles) if a.profiles else None) if a.command=='plan' else report(read(a.paper),a.root,read(a.directories),read(a.references))
 write_json(a.output,r)
