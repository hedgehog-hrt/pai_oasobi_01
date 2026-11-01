"""Build a source release, preview SSH operations, deploy, initialize runs and retrieve logs.
All network commands require --execute. This does not rent/start/stop a Pod or install packages.
"""
import argparse,json,os,re,shlex,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro import __version__
from rosa_repro import pod_transfer as transfer

RELEASE_TOP=('src','scripts','tests','docs','examples')
RELEASE_FILES=('README.md','RESULTS.md','pyproject.toml','requirements.lock','requirements-serve-cpu.lock','requirements-vlm-cpu.lock','requirements-vlm-gpu.in')


def source_bundle(root,output):
    root=Path(root).resolve();paths=[]
    for folder in RELEASE_TOP:
        for p in sorted((root/folder).rglob('*')):
            if any(x in ('__pycache__','.pytest_cache') or x.startswith('.') for x in p.relative_to(root).parts):continue
            if p.is_symlink():raise ValueError('Release source contains a symlink')
            if p.is_file():
                if p.suffix not in ('.py','.json','.md','.sh','.toml'):raise ValueError('Unexpected release file type: '+str(p))
                paths.append(p)
    paths.extend(root/n for n in RELEASE_FILES)
    if sum(p.stat().st_size for p in paths)>50*1024**2:raise ValueError('Source release exceeds 50 MiB; model/data/log uploads are separate')
    return transfer.pack(root,paths,output,kind='release',context=dict(project='P-AI ROSA',version=__version__,scope='code/config/docs/tests/requirements only; no models, caches, credentials or prior results'))


def connection(c,*,ready=True):
    if set(c)!={'schema_version','host','port','user','identity_file','remote_base','remote_python'} or c['schema_version']!='1.0':raise ValueError('Invalid connection keys/version')
    if not isinstance(c['host'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]{0,252}',c['host']):raise ValueError('Use an IPv4 address or hostname')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*',c['user']):raise ValueError('Invalid SSH user')
    if type(c['port']) is not int or not 1<=c['port']<=65535:raise ValueError('Invalid SSH port')
    base=Path(c['remote_base'])
    if not base.is_absolute() or len(base.parts)<3 or not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(base)) or '..' in base.parts:raise ValueError('Use a dedicated plain absolute remote base')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',c['remote_python']) or '..' in Path(c['remote_python']).parts:raise ValueError('Use a plain absolute remote Python path')
    key=Path(c['identity_file']).expanduser()
    if not key.is_absolute():raise ValueError('SSH identity path must be absolute or start with ~/')
    if ready and ('SET_' in json.dumps(c) or c['host'].startswith('SET-')):raise ValueError('Resolve the connection template before executing')
    return dict(c,identity_file=str(key))


def ssh_argv(c,operation,arguments):
    c=connection(c);program=Path(transfer.__file__).read_text()
    remote=[c['remote_python'],'-c',program,operation,'--base',c['remote_base'],*map(str,arguments)]
    return ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-p',str(c['port']),'-i',c['identity_file'],
        c['user']+'@'+c['host'],shlex.join(remote)]


class SSH:
    def __init__(self,c,timeout=600):self.c=c;self.timeout=timeout
    def call(self,op,args=(),*,stdin=None,stdout=None):
        r=subprocess.run(ssh_argv(self.c,op,args),stdin=stdin,stdout=stdout or subprocess.PIPE,stderr=subprocess.PIPE,timeout=self.timeout,check=False)
        if r.returncode:raise RuntimeError('SSH '+op+' failed: '+r.stderr.decode(errors='replace')[-4000:])
        return json.loads(r.stdout) if stdout is None else None


def deploy(client,bundle):
    m=transfer.validate_archive(bundle,expected_kind='release');h=transfer.sha(bundle)
    with Path(bundle).open('rb') as f:r=client.call('receive',['--sha',h,'--size',Path(bundle).stat().st_size],stdin=f)
    if r['bundle_id']!=m['bundle_id'] or r['status'] not in ('installed','already_present'):raise ValueError('Deployment receipt differs')
    return dict(r,archive_sha256=h)


def collect(client,run_id,output):
    transfer.identifier(run_id);output=transfer.safe_path(output);output.mkdir(parents=True,exist_ok=True)
    r=client.call('snapshot',['--run-id',run_id]);transfer.hash_id(r['archive_sha256']);transfer.hash_id(r['bundle_id'])
    if r['run_id']!=run_id or type(r['bytes']) is not int or not 1<=r['bytes']<=transfer.MAX_BYTES:raise ValueError('Invalid snapshot receipt')
    destination=output/run_id/r['bundle_id'];partial=output/(r['archive_sha256']+'.part')
    # Reuse only an exactly matching completed local extraction, otherwise transfer anew.
    if destination.exists():
        m=json.loads((transfer.safe_path(destination)/transfer.MANIFEST).read_text());transfer.verify_installed(destination,m)
        if m['bundle_id']!=r['bundle_id'] or m['context']['run']['run_id']!=run_id:raise ValueError('Local snapshot identity mismatch')
        return dict(r,status='already_present',directory=str(destination))
    transfer.safe_path(partial)
    if partial.exists() and partial.stat().st_nlink!=1:raise ValueError('Hardlinked local staging file')
    with partial.open('wb') as f:client.call('download',['--sha',r['archive_sha256']],stdout=f)
    if partial.stat().st_size!=r['bytes'] or transfer.sha(partial)!=r['archive_sha256']:raise ValueError('Downloaded archive changed/truncated; staging retained')
    m=transfer.validate_archive(partial,expected_kind='logs')
    if m['bundle_id']!=r['bundle_id'] or m['context']['run']['run_id']!=run_id:raise ValueError('Downloaded run identity mismatch')
    result=transfer.unpack(partial,destination,expected_kind='logs');partial.unlink()
    return dict(r,**result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('template');q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('pack');q.add_argument('--root',type=Path,default=ROOT);q.add_argument('--output',type=Path,required=True)
    for name in ('status','deploy','init-run','collect'):
        q=sub.add_parser(name);q.add_argument('--connection',type=Path,required=True);q.add_argument('--execute',action='store_true')
        q.add_argument('--report',type=Path,required=True);q.add_argument('--timeout',type=int,default=600)
        if name=='deploy':q.add_argument('--bundle',type=Path,required=True)
        if name in ('init-run','collect'):q.add_argument('--run-id',required=True)
        if name=='init-run':q.add_argument('--release-id',required=True)
        if name=='collect':q.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='template':
        result=dict(schema_version='1.0',host='SET-HOST',port=22,user='root',identity_file='~/.ssh/id_ed25519',remote_base='/workspace/p-ai-rosa',remote_python='/usr/bin/python3')
    elif a.command=='pack':result=source_bundle(a.root,a.output);print(json.dumps(dict(bundle_id=result['bundle_id'],archive_sha256=transfer.sha(a.output),files=len(result['files']))));return 0
    else:
        if a.report.exists():raise FileExistsError(a.report)
        c=connection(json.loads(a.connection.read_text()),ready=a.execute)
        if not 1<=a.timeout<=7200:raise ValueError('Transfer timeout out of range')
        result=dict(operation=a.command,dry_run=not a.execute,connection=c,remote_helper_sha256=transfer.sha(transfer.__file__),
            strict_host_key_checking=True,remote_execution_performed=False,pod_lifecycle_operations=False)
        if a.command in ('init-run','collect'):transfer.identifier(a.run_id);result['run_id']=a.run_id
        if a.command=='init-run':transfer.hash_id(a.release_id);result['release_id']=a.release_id
        if a.command=='deploy':
            m=transfer.validate_archive(a.bundle,expected_kind='release');result.update(bundle_id=m['bundle_id'],archive_sha256=transfer.sha(a.bundle),remote_destination=c['remote_base']+'/releases/'+m['bundle_id'])
        code=0
        try:
            if a.execute:
                client=SSH(c,a.timeout);result['remote_execution_performed']=True
                if a.command=='status':result['receipt']=client.call('status')
                elif a.command=='deploy':result['receipt']=deploy(client,a.bundle)
                elif a.command=='init-run':result['receipt']=client.call('init-run',['--run-id',a.run_id,'--release-id',a.release_id])
                else:result['receipt']=collect(client,a.run_id,a.output)
                result['status']='complete'
            else:result['status']='planned'
        except Exception as exc:result.update(status='failed',error_type=type(exc).__name__,error=str(exc));code=1
        a.report.parent.mkdir(parents=True,exist_ok=True)
        with a.report.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
        print(json.dumps(dict(status=result['status'],report=str(a.report),dry_run=result['dry_run'])))
        return code
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
