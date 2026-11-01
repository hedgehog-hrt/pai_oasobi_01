"""Export pinned T01 states or run paired regression in the existing sim env."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.regression import verify_upstream
from rosa_repro.libero_bridge import export_t01_bank, LiberoEpisode, ZmqRegressionTransport
from rosa_repro.paired import run_paired,capture_initial_observations,load_state_bank
from rosa_repro.r06_protocol import TASKS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['export-states','capture-observations','run'])
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',type=Path)
    p.add_argument('--port',type=int,default=5556)
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--task',choices=list(TASKS),default='T01')
    args=p.parse_args()
    verify_upstream(args.repo)
    sys.path.insert(0,str(args.repo.resolve()))
    if args.command=='export-states':
        export_t01_bank(args.output,args.count,task_name=TASKS[args.task])
        return 0
    if args.bank is None:
        p.error('--bank required for run')
    manifest,_=load_state_bank(args.bank)
    if manifest['task'] not in TASKS.values():p.error('Unknown LIBERO task in state bank')
    from gr00t.policy.server_client import PolicyClient
    with PolicyClient(host='127.0.0.1',port=args.port) as client:
        configs=client.get_modality_config()
    if args.command=='capture-observations':
        capture_initial_observations(args.bank,args.output,lambda:LiberoEpisode(configs,task_name=manifest['task']))
        return 0
    result=run_paired(bank=args.bank,output=args.output,env_factory=lambda:LiberoEpisode(configs,task_name=manifest['task']),
        transport_factory=lambda arm:ZmqRegressionTransport(arm,args.port),execution_kind='libero_gpu')
    return 0 if result['paired_comparison_valid'] else 1


if __name__=='__main__':
    raise SystemExit(main())
