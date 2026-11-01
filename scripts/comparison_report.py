"""Recompute declared comparison cells and preserve missing/failed repetitions."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.comparison_report import report,markdown,condition
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('condition');c.add_argument('directory',type=Path);c.add_argument('--output',type=Path,required=True)
    c=s.add_parser('build');c.add_argument('--manifest',type=Path,required=True);c.add_argument('--root',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.command=='condition':write_json(a.output,condition(a.directory));return 0
    r=report(read(a.manifest),a.root);a.output.mkdir(parents=True)
    write_json(a.output/'report.json',r);(a.output/'report.md').write_text(markdown(r))
    return 0 if r['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
