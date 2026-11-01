"""Inspect raw GPU calibration repetitions and create an unapproved review template."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.dedicated_residency import inspect,template
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--directories',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 c=inspect(read(a.plan),a.root,read(a.directories));a.output.mkdir(parents=True)
 write_json(a.output/'candidate.json',c);write_json(a.output/'review-template.json',template(c))
