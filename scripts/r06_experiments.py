"""Prepare/check R06 or run exactly one explicitly selected cell in a supervised process."""
import argparse,asyncio,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.r06_protocol import prepare,validate
from rosa_repro.r06_runtime import run_cell,reaudit
from rosa_repro.regression import write_json
from rosa_repro.resume_workflow import supervise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','check','run','_trial','reaudit'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bank-root',type=Path);p.add_argument('--suite',type=Path)
    p.add_argument('--mode',choices=['cpu_dummy','libero_gpu'],default='cpu_dummy');p.add_argument('--phase',choices=['smoke','formal'],default='smoke')
    p.add_argument('--selected-batch',type=int,default=4);p.add_argument('--server-evidence',type=Path)
    p.add_argument('--cell');p.add_argument('--sim-python');p.add_argument('--sim-repo',type=Path);p.add_argument('--port',type=int,default=5556)
    p.add_argument('--execute',action='store_true');p.add_argument('--trial-timeout-s',type=int,default=7200)
    p.add_argument('--raw',type=Path);a=p.parse_args();read=lambda path:json.loads(path.read_text())
    if a.command=='reaudit':write_json(a.output,reaudit(a.raw));return 0
    if a.command=='prepare':
        write_json(a.output,prepare(a.bank_root,mode=a.mode,phase=a.phase,selected_batch=a.selected_batch,
            server_evidence=read(a.server_evidence) if a.server_evidence else None,trial_timeout_s=a.trial_timeout_s));return 0
    s=validate(read(a.suite),a.bank_root)
    if a.command=='check':write_json(a.output,dict(status='valid',suite_sha256=s['sha256'],cells=s['cells'],executed=False));return 0
    if a.cell not in {c['id'] for c in s['cells']}:p.error('--cell must identify exactly one suite cell')
    if a.command=='run' and not a.execute:
        write_json(a.output,dict(status='planned',cell=a.cell,executed=False,wall_budget_s=s['budgets']['trial_timeout_s']+2*s['budgets']['call_timeout_s']+30));return 0
    if not a.sim_python or not Path(a.sim_python).is_absolute() or not Path(a.sim_python).is_file():p.error('Explicit simulator Python required')
    if (s['mode']=='libero_gpu')!=(a.sim_repo is not None):p.error('Pinned simulator repo required only for GPU mode')
    if a.command=='_trial':
        r=asyncio.run(run_cell(s,a.bank_root,a.cell,a.output,sim_python=a.sim_python,sim_repo=a.sim_repo,port=a.port))
        return 0 if r['status']=='complete' else 2
    a.output.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(Path(__file__).resolve()),'_trial','--suite',str(a.suite.resolve()),'--bank-root',str(a.bank_root.resolve()),
        '--cell',a.cell,'--output',str(a.output.resolve()/'raw'),'--sim-python',a.sim_python,'--port',str(a.port)]
    if a.sim_repo:cmd+=['--sim-repo',str(a.sim_repo.resolve())]
    budget=s['budgets']['trial_timeout_s']+2*s['budgets']['call_timeout_s']+30
    write_json(a.output/'command.json',dict(argv=cmd,wall_budget_s=budget))
    process=supervise(cmd,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),a.output,budget)
    audit=reaudit(a.output/'raw') if (a.output/'raw/evidence.json').exists() else None
    passed=process['exit_code']==0 and audit and audit['status']=='complete'
    write_json(a.output/'supervision.json',dict(process=process,audit=audit,status='complete' if passed else 'failed',
        server_inflight_execution_unknown=s['mode']=='libero_gpu' and not passed))
    return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
