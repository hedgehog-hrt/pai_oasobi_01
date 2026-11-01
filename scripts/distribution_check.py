"""Check shipped configuration examples without network access or model loading."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.distribution import check
from rosa_repro.regression import write_json
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=check(a.root);write_json(a.output,r);print(json.dumps(r))
