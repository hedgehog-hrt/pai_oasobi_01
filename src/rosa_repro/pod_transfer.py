"""Standard-library-only immutable release/log bundles and remote operations.
This exact source is sent to the user's selected SSH host; no dependency install.
"""
import argparse,contextlib,fcntl,hashlib,json,os,re,shutil,stat,sys,tempfile,zipfile
from pathlib import Path,PurePosixPath

MANIFEST='ROSA_BUNDLE.json'
MAX_BYTES=20*1024**3
MAX_FILES=50000


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def relpath(name):
    p=PurePosixPath(name)
    if not isinstance(name,str) or not name or len(name)>1024 or '\\' in name or any(ord(x)<32 for x in name) or p.is_absolute() or any(x in ('.','..','') for x in name.split('/')) or ':' in name:
        raise ValueError('Unsafe archive relative path')
    return p


def identifier(x):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',x):raise ValueError('Invalid run ID')
    return x


def hash_id(x):
    if not re.fullmatch('[0-9a-f]{64}',x):raise ValueError('Invalid SHA256 ID')
    return x


def safe_path(path):
    """Reject symlink ancestors, even when a symlink stays inside the root."""
    path=Path(os.path.abspath(path))
    for p in (path,*path.parents):
        if p.is_symlink():raise ValueError('Symlink path is not permitted')
    return path


def identity(path):
    s=path.stat();return (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def all_files(root):
    root=safe_path(root);paths=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Symlink in selected tree')
        if p.is_dir():continue
        if not p.is_file():raise ValueError('Special file in selected tree')
        if p.name in ('.runner.lock','.rosa-transfer.lock'):continue
        if p.name.endswith(('-wal','-shm','-journal')):
            if p.stat().st_size:raise ValueError('SQLite sidecar present; stop/checkpoint the writer before collecting')
            continue
        paths.append(p)
    return paths


def pack(root,paths,output,*,kind,context):
    root=safe_path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if kind not in ('release','logs'):raise ValueError('Unknown bundle kind')
    entries={};before={}
    for p in paths:
        p=safe_path(p)
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_nlink!=1:raise ValueError('File outside root/nonregular/hardlinked')
        name=p.relative_to(root).as_posix();relpath(name)
        if name==MANIFEST or name in entries:raise ValueError('Reserved/duplicate bundle path')
        before[name]=identity(p);entries[name]=dict(size=p.stat().st_size,sha256=sha(p))
    if not entries or len(entries)>MAX_FILES or sum(x['size'] for x in entries.values())>MAX_BYTES:raise ValueError('Bundle size/count exceeds bounds')
    if len({x.casefold() for x in entries})!=len(entries):raise ValueError('Case-colliding bundle paths')
    m=dict(schema_version='1.0',kind=kind,context=context,files=entries);m['bundle_id']=digest(m)
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
            header=zipfile.ZipInfo(MANIFEST);header.compress_type=zipfile.ZIP_DEFLATED;header.external_attr=(stat.S_IFREG|0o600)<<16
            z.writestr(header,canonical(m))
            for name in sorted(entries):
                # Copy files as bytes, never preserve symlink/device attributes.
                info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o600)<<16
                with (root/name).open('rb') as f,z.open(info,'w') as out:shutil.copyfileobj(f,out,4*1024**2)
                if identity(root/name)!=before[name]:raise ValueError('Source file changed during snapshot')
        validate_archive(output)
    except BaseException:
        # Only this function's new, incomplete archive is removed.
        output.unlink(missing_ok=True);raise
    return m


def validate_archive(archive,*,expected_kind=None):
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)>MAX_FILES+1 or len(names)!=len(set(names)) or len({x.casefold() for x in names})!=len(names):raise ValueError('Duplicate/excess archive entries')
        for i in infos:
            relpath(i.filename);mode=i.external_attr>>16
            if i.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0,stat.S_IFREG) or i.flag_bits&1:raise ValueError('Only unencrypted regular files accepted')
        if MANIFEST not in names or z.getinfo(MANIFEST).file_size>8*1024**2:raise ValueError('Missing/oversized bundle manifest')
        m=json.loads(z.read(MANIFEST));base=dict(m);base.pop('bundle_id',None)
        if m.get('schema_version')!='1.0' or m.get('kind') not in ('release','logs') or m.get('bundle_id')!=digest(base):raise ValueError('Bundle manifest hash/version mismatch')
        if expected_kind and m['kind']!=expected_kind:raise ValueError('Wrong bundle kind')
        if not m['files'] or set(names)!={MANIFEST,*m['files']}:raise ValueError('Archive/manifest coverage mismatch')
        total=0
        for name,row in m['files'].items():
            if set(row)!={'size','sha256'} or type(row['size']) is not int or row['size']<0:raise ValueError('Invalid manifest file metadata')
            hash_id(row['sha256']);total+=row['size']
            if total>MAX_BYTES or z.getinfo(name).file_size!=row['size']:raise ValueError('Archive size mismatch/limit')
            h=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
            if h.hexdigest()!=row['sha256']:raise ValueError('Archive content hash mismatch')
    return m


def verify_installed(directory,manifest):
    base=dict(manifest);base.pop('bundle_id',None)
    if manifest.get('bundle_id')!=digest(base) or manifest.get('kind') not in ('release','logs') or manifest.get('schema_version')!='1.0':
        raise ValueError('Installed manifest hash/version mismatch')
    for name in manifest['files']:relpath(name)
    directory=safe_path(directory)
    files={p.relative_to(directory).as_posix() for p in all_files(directory)}
    if files!={MANIFEST,*manifest['files']}:raise ValueError('Installed tree differs from manifest')
    if json.loads((directory/MANIFEST).read_text())!=manifest:raise ValueError('Installed manifest mismatch')
    for name,row in manifest['files'].items():
        p=directory/name
        if p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:raise ValueError('Installed content changed')
    return True


def unpack(archive,destination,*,expected_kind=None):
    m=validate_archive(archive,expected_kind=expected_kind);destination=safe_path(destination)
    if destination.exists():verify_installed(destination,m);return dict(status='already_present',bundle_id=m['bundle_id'],directory=str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.rosa-unpack-',dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            for name in [MANIFEST,*m['files']]:
                p=temp/name;p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as out,z.open(name) as src:shutil.copyfileobj(src,out,4*1024**2)
                p.chmod(0o600 if m['kind']=='logs' else 0o644)
        verify_installed(temp,m)
        if destination.exists():raise FileExistsError(destination)
        temp.rename(destination)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return dict(status='installed',bundle_id=m['bundle_id'],directory=str(destination))


def layout(base):
    base=safe_path(base)
    if len(base.parts)<3:raise ValueError('Use a dedicated absolute multi-component base directory')
    marker=base/'.rosa-layout.json'
    if base.exists() and not marker.exists() and any(base.iterdir()):raise ValueError('Refuse unmanaged nonempty base directory')
    base.mkdir(parents=True,exist_ok=True)
    value=dict(schema_version='1.0',purpose='ROSA isolated releases and experiment artifacts')
    if marker.exists():
        safe_path(marker)
        if json.loads(marker.read_text())!=value:raise ValueError('Layout marker differs')
    else:marker.write_bytes(canonical(value))
    for name in ('incoming','releases','runs','exports'):safe_path(base/name).mkdir(exist_ok=True)
    return base


@contextlib.contextmanager
def lock(path):
    path=safe_path(path)
    with path.open('a+b') as f:
        if path.stat().st_nlink!=1:raise ValueError('Hardlinked lock file')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Operation/run is busy')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def receive(base,archive_sha,size,stream):
    base=layout(base);hash_id(archive_sha)
    if not 1<=size<=MAX_BYTES:raise ValueError('Upload size exceeds limit')
    with lock(base/'.rosa-transfer.lock'):
        partial=safe_path(base/'incoming'/(archive_sha+'.part'))
        if partial.exists() and partial.stat().st_nlink!=1:raise ValueError('Hardlinked incoming file')
        # Restart interrupted transfer in this owned staging file. Final releases are never overwritten.
        with partial.open('wb') as out:
            remaining=size
            while remaining:
                block=stream.read(min(4*1024**2,remaining))
                if not block:raise ValueError('Interrupted upload')
                out.write(block);remaining-=len(block)
            out.flush();os.fsync(out.fileno())
        if sha(partial)!=archive_sha:raise ValueError('Upload SHA256 mismatch')
        m=validate_archive(partial,expected_kind='release')
        result=unpack(partial,base/'releases'/m['bundle_id'],expected_kind='release')
        partial.unlink();return result


def init_run(base,run_id,release_id):
    base=layout(base);identifier(run_id);hash_id(release_id)
    with lock(base/'.rosa-transfer.lock'):
        release=safe_path(base/'releases'/release_id);m=json.loads((release/MANIFEST).read_text());verify_installed(release,m)
        if m['bundle_id']!=release_id or m['kind']!='release':raise ValueError('Release identity differs from directory')
        p=safe_path(base/'runs'/run_id);value=dict(schema_version='1.0',run_id=run_id,release_id=release_id)
        if p.exists():
            if json.loads((safe_path(p/'.rosa-run.json')).read_text())!=value:raise ValueError('Run ID already bound to another release')
        else:p.mkdir();(p/'.rosa-run.json').write_bytes(canonical(value))
        return dict(status='ready',directory=str(p),release=str(release),run=value)


def snapshot(base,run_id):
    base=layout(base);identifier(run_id);root=safe_path(base/'runs'/run_id)
    run=json.loads(safe_path(root/'.rosa-run.json').read_text())
    with lock(base/'.rosa-transfer.lock'),contextlib.ExitStack() as stack:
        for p in sorted(root.rglob('.runner.lock')):stack.enter_context(lock(p))
        paths=all_files(root);before={p.relative_to(root).as_posix():identity(p) for p in paths}
        staged=base/'exports'/('snapshot-'+run_id+'.part')
        if staged.exists():raise ValueError('Unfinished snapshot staging exists; preserve/inspect before retry')
        try:
            m=pack(root,paths,staged,kind='logs',context=dict(run=run,scope='stable snapshot; does not certify experiment completion'))
            after={p.relative_to(root).as_posix():identity(p) for p in all_files(root)}
            if before!=after:raise ValueError('Run tree changed during collection')
            archive_sha=sha(staged);dest=base/'exports'/(archive_sha+'.zip')
            if dest.exists():
                safe_path(dest)
                if sha(dest)!=archive_sha:raise ValueError('Existing export was changed')
                staged.unlink()
            else:staged.rename(dest)
        except BaseException:staged.unlink(missing_ok=True);raise
        return dict(archive_sha256=archive_sha,bytes=dest.stat().st_size,bundle_id=m['bundle_id'],run_id=run_id,files=len(m['files']))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','receive','init-run','snapshot','download']);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--sha');p.add_argument('--size',type=int);p.add_argument('--run-id');p.add_argument('--release-id');a=p.parse_args()
    if a.command=='status':
        # Read only: a status request never creates a layout.
        base=safe_path(a.base);result=dict(base_exists=base.exists(),python=sys.version.split()[0],base=str(base))
    elif a.command=='receive':result=receive(a.base,a.sha,a.size,sys.stdin.buffer)
    elif a.command=='init-run':result=init_run(a.base,a.run_id,a.release_id)
    elif a.command=='snapshot':result=snapshot(a.base,a.run_id)
    else:
        hash_id(a.sha);p=safe_path(a.base/'exports'/(a.sha+'.zip'))
        if sha(p)!=a.sha:raise ValueError('Export content hash changed')
        with p.open('rb') as f:shutil.copyfileobj(f,sys.stdout.buffer,4*1024**2)
        return
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
