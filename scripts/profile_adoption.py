"""Inspect, review and import measured profiles without starting models."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.profile_adoption import inspect,review_template,adopt,validate_adoption,import_profiles,read
from rosa_repro.regression import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['inspect','review-template','adopt','validate','import'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--request',type=Path);p.add_argument('--review',type=Path)
    p.add_argument('--adoptions',type=Path,nargs='+');p.add_argument('--fleet',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    try:
        if a.command in ('inspect','review-template','adopt'):
            if not a.request:p.error('--request required')
            if a.command=='adopt':
                if not a.review:p.error('--review required')
                result=adopt(read(a.request),a.root,read(a.review))
            else:
                result=inspect(read(a.request),a.root)
                if a.command=='review-template':result=review_template(result)
        else:
            if not a.adoptions:p.error('--adoptions required')
            values=[validate_adoption(read(x),a.root) for x in a.adoptions]
            if a.command=='import':
                if not a.fleet:p.error('--fleet required')
                result=import_profiles(read(a.fleet),values,a.root)
            else:result={'status':'passed','adoption_ids':[x['adoption_id'] for x in values]}
    except (ValueError,KeyError,TypeError,OSError) as exc:
        write_json(a.output,dict(status='blocked',error_type=type(exc).__name__,reason=str(exc)));return 2
    write_json(a.output,result);return 0
if __name__=='__main__':raise SystemExit(main())
