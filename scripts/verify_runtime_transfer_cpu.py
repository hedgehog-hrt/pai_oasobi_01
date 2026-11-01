"""Deploy a real source bundle to a local simulated Pod, run CPU diagnostics, retrieve logs.
Uses the exact remote helper in subprocesses. Does not use SSH or a network.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro import pod_transfer as transfer
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.profile_runner import verify_evidence
from pod_sync import deploy,collect,main as sync_main


class LocalRemote:
    def __init__(self,base):self.base=base;self.operations=[]
    def call(self,op,args=(),*,stdin=None,stdout=None):
        self.operations.append(op)
        r=subprocess.run([sys.executable,str(Path(transfer.__file__).resolve()),op,'--base',str(self.base),*map(str,args)],
            stdin=stdin,stdout=stdout or subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
        return json.loads(r.stdout) if stdout is None else None


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def verify(bundle,out):
    out.mkdir(parents=True,exist_ok=False);m=transfer.validate_archive(bundle,expected_kind='release')
    client=LocalRemote(out/'simulated-pod');first=deploy(client,bundle);second=deploy(client,bundle)
    assert first['status']=='installed' and second['status']=='already_present'
    run=client.call('init-run',['--run-id','cpu-diagnostic','--release-id',m['bundle_id']]);run_dir=Path(run['directory']);release=Path(run['release'])
    env=dict(os.environ,PYTHONPATH=str(release/'src'),PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(release/'scripts/runtime_evidence.py'),'probe','--cpu-self-test','--timeout','60','--output',str(run_dir/'kernel')]
    subprocess.run(command,env=env,cwd=release,check=True,timeout=75,capture_output=True)
    verify_evidence(run_dir/'kernel');result=json.loads((run_dir/'kernel/result.json').read_text())
    assert result['status']=='complete' and not result['gpu_execution'] and not result['gpu_kernel_events_present']
    assert result['traces']['cpu-trace.json']['cpu_operator_names']
    assert json.loads((run_dir/'kernel/source_manifest.json').read_text())==runtime_source_manifest()
    # Execution must leave the immutable code tree byte-for-byte intact.
    transfer.verify_installed(release,m)
    received=collect(client,'cpu-diagnostic',out/'retrieved');again=collect(client,'cpu-diagnostic',out/'retrieved')
    assert received['status']=='installed' and again['status']=='already_present'
    recovered=Path(received['directory']);verify_evidence(recovered/'kernel')
    assert transfer.sha(recovered/'kernel/cpu-trace.json')==transfer.sha(run_dir/'kernel/cpu-trace.json')
    c=dict(schema_version='1.0',host='SET-HOST',port=22,user='root',identity_file='~/.ssh/id_ed25519',
        remote_base='/workspace/p-ai-rosa',remote_python='/usr/bin/python3')
    write(out/'connection-unresolved.json',c)
    assert sync_main(['deploy','--connection',str(out/'connection-unresolved.json'),'--bundle',str(bundle),'--report',str(out/'deployment-dry-run.json')])==0
    report=json.loads((out/'deployment-dry-run.json').read_text());assert not report['remote_execution_performed']
    for name,value in [('deploy-first.json',first),('deploy-repeat.json',second),('run-init.json',run),('collection.json',received),('collection-repeat.json',again)]:write(out/name,value)
    verified=dict(status='passed',source_manifest=runtime_source_manifest(),release_bundle_id=m['bundle_id'],release_archive_sha256=transfer.sha(bundle),
        source_release_installed_verified=True,idempotent_deployment=True,immutable_source_after_execution=True,
        real_torch_CPU_trace_collected=True,cpu_trace_does_not_claim_GPU=True,downloaded_raw_trace_hash_identical=True,
        idempotent_collection=True,unresolved_connection_dry_run_no_network=True,remote_helper_operations=client.operations,
        ssh_authentication_tested=False,network_used=False,pod_access_attempted=False,gpu_execution=False,pretrained_weights_loaded=False,
        scope='Local subprocess simulation of transfer protocol plus actual CPU Torch profiler, not SSH/GPU validation')
    write(out/'verification.json',verified);print(json.dumps({k:v for k,v in verified.items() if k!='source_manifest'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    verify(a.bundle.resolve(),a.output.resolve())
