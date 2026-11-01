"""Prepare/check/run a bounded local resumption workflow; never SSH or start a Pod."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.resume_workflow import prepare,check,run
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--config',required=True);a.add_argument('--corpus-root',required=True)
    for stage in ('smoke','pilot','profile'):a.add_argument('--'+stage+'-suite',required=stage=='smoke')
    a.add_argument('--driver-python',required=True);a.add_argument('--worker-python',required=True);a.add_argument('--release')
    a.add_argument('--setup-budget',type=float,default=600);a.add_argument('--kernel-budget',type=int,default=600);a.add_argument('--diagnostic-device')
    a.add_argument('--output',type=Path,required=True)
    for name in ('check','run'):
        a=s.add_parser(name);a.add_argument('--plan',type=Path,required=True);a.add_argument('--report',type=Path,required=True)
        if name=='run':
            a.add_argument('--output',type=Path,required=True);a.add_argument('--execute',action='store_true');a.add_argument('--max-new',type=int,default=1)
            a.add_argument('--budget-seconds',type=float);a.add_argument('--address');a.add_argument('--cleanup-review',type=Path)
    a=p.parse_args();target=a.output if a.command=='prepare' else a.report
    if target.exists():raise FileExistsError(target)
    try:
        if a.command=='prepare':result=prepare(a.config,a.corpus_root,{k:getattr(a,k+'_suite') for k in ('smoke','pilot','profile') if getattr(a,k+'_suite')},
            driver_python=a.driver_python,worker_python=a.worker_python,release=a.release,setup_budget_s=a.setup_budget,kernel_budget_s=a.kernel_budget,diagnostic_device=a.diagnostic_device)
        elif a.command=='check':result=check(read(a.plan))
        else:result=run(read(a.plan),a.output,execute=a.execute,max_new=a.max_new,budget_s=a.budget_seconds,address=a.address,
                        cleanup_review=read(a.cleanup_review) if a.cleanup_review else None)
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        result=dict(status='blocked',error_type=type(exc).__name__,reason=str(exc))
    write_json(target,result);print(json.dumps({k:v for k,v in result.items() if k in ('status','reason','plan_id','executed_steps')}))
    return 0 if result.get('status') in (None,'ready','planned','complete','step_limit') else 2
if __name__=='__main__':raise SystemExit(main())
