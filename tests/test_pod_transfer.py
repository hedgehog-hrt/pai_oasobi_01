import fcntl,importlib.util,io,json,os,shlex,stat,subprocess,sys,zipfile
from pathlib import Path
import pytest
from rosa_repro import pod_transfer as t
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pod_sync_tests',ROOT/'scripts/pod_sync.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def bundle(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'code.py').write_text('print(42)\n')
    zip_path=tmp_path/'release.zip';m=t.pack(root,[root/'code.py'],zip_path,kind='release',context={'version':'synthetic-test'})
    return zip_path,m


class LocalRemote:
    """Execute the identical stdin/JSON/binary remote helper in a subprocess, without SSH."""
    def __init__(self,base):self.base=base;self.calls=[]
    def call(self,op,args=(),*,stdin=None,stdout=None):
        self.calls.append(op)
        r=subprocess.run([sys.executable,str(Path(t.__file__)),op,'--base',str(self.base),*map(str,args)],
            stdin=stdin,stdout=stdout or subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        return json.loads(r.stdout) if stdout is None else None


def test_complete_local_transport_deploy_retry_run_and_collection(tmp_path):
    z,m=bundle(tmp_path);client=LocalRemote(tmp_path/'remote')
    a=cli.deploy(client,z);b=cli.deploy(client,z)
    assert a['status']=='installed' and b['status']=='already_present'
    run=client.call('init-run',['--run-id','test-1','--release-id',m['bundle_id']]);r=Path(run['directory'])
    (r/'events.jsonl').write_text('{"event":"synthetic"}\n');(r/'raw.bin').write_bytes(bytes(range(256)))
    receipt=cli.collect(client,'test-1',tmp_path/'download');assert receipt['status']=='installed'
    destination=Path(receipt['directory']);assert (destination/'raw.bin').read_bytes()==(r/'raw.bin').read_bytes()
    receipt2=cli.collect(client,'test-1',tmp_path/'download');assert receipt2['bundle_id']==receipt['bundle_id'] and receipt2['status']=='already_present'
    (r/'later.json').write_text('{}');receipt3=cli.collect(client,'test-1',tmp_path/'download')
    assert receipt3['bundle_id']!=receipt['bundle_id'] and destination.exists()


def test_interrupted_upload_is_staged_and_retry_preserves_release(tmp_path):
    z,m=bundle(tmp_path);base=tmp_path/'remote'
    with pytest.raises(ValueError,match='Interrupted'):t.receive(base,t.sha(z),z.stat().st_size,io.BytesIO(z.read_bytes()[:30]))
    assert not list((base/'releases').iterdir()) and list((base/'incoming').glob('*.part'))
    with z.open('rb') as f:assert t.receive(base,t.sha(z),z.stat().st_size,f)['status']=='installed'
    target=base/'releases'/m['bundle_id'];(target/'code.py').write_text('changed')
    with z.open('rb') as f,pytest.raises(ValueError):t.receive(base,t.sha(z),z.stat().st_size,f)
    assert (target/'code.py').read_text()=='changed'


def test_changed_installed_manifest_cannot_authorize_changed_source(tmp_path):
    z,m=bundle(tmp_path);out=tmp_path/'installed';t.unpack(z,out)
    (out/'code.py').write_text('changed');m['files']['code.py']={'size':7,'sha256':t.sha(out/'code.py')};(out/t.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(ValueError,match='manifest hash'):t.verify_installed(out,m)


@pytest.mark.parametrize('bad',['traversal','absolute','symlink','duplicate','corrupt','extra','case_collision'])
def test_archive_validation_rejects_unsafe_or_unbound_contents(tmp_path,bad):
    z,m=bundle(tmp_path);target=tmp_path/'bad.zip'
    with zipfile.ZipFile(z) as source,zipfile.ZipFile(target,'w') as out:
        for info in source.infolist():
            value=source.read(info.filename)
            if bad=='corrupt' and info.filename=='code.py':value=b'different'
            out.writestr(info,value)
        if bad=='traversal':out.writestr('../outside','x')
        if bad=='absolute':out.writestr('/tmp/outside','x')
        if bad=='extra':out.writestr('unbound','x')
        if bad=='duplicate':out.writestr('code.py','x')
        if bad=='case_collision':out.writestr('CODE.PY','x')
        if bad=='symlink':
            info=zipfile.ZipInfo('link');info.external_attr=(stat.S_IFLNK|0o777)<<16;out.writestr(info,'../elsewhere')
    with pytest.raises(ValueError):t.unpack(target,tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()


def test_busy_run_and_SQLite_sidecars_block_collection(tmp_path):
    z,m=bundle(tmp_path);base=tmp_path/'remote'
    with z.open('rb') as f:t.receive(base,t.sha(z),z.stat().st_size,f)
    r=Path(t.init_run(base,'r1',m['bundle_id'])['directory']);lock=r/'.runner.lock'
    with lock.open('wb') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='busy'):t.snapshot(base,'r1')
    (r/'store.sqlite-wal').write_bytes(b'nonempty')
    with pytest.raises(ValueError,match='SQLite'):t.snapshot(base,'r1')


def test_symlinked_log_cannot_collect_external_files(tmp_path):
    root=tmp_path/'run';root.mkdir();outside=tmp_path/'outside';outside.write_text('external')
    (root/'link').symlink_to(outside)
    with pytest.raises(ValueError):t.all_files(root)
    with pytest.raises(ValueError):t.pack(root,[root/'link'],tmp_path/'never.zip',kind='logs',context={})


def test_source_changed_during_pack_is_not_published(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();p=root/'events.jsonl';p.write_text('before');original=t.sha
    def changed(file):
        value=original(file)
        if Path(file)==p:p.write_text('after')
        return value
    monkeypatch.setattr(t,'sha',changed)
    with pytest.raises(ValueError):t.pack(root,[p],tmp_path/'new.zip',kind='logs',context={})
    assert not (tmp_path/'new.zip').exists()


def connection():return dict(schema_version='1.0',host='pod.example.invalid',port=22,user='root',identity_file='/tmp/test key',remote_base='/workspace/p-ai-rosa',remote_python='/usr/bin/python3')


def test_default_dry_run_never_invokes_subprocess_or_reads_key(tmp_path,monkeypatch):
    z,m=bundle(tmp_path);c=tmp_path/'connection.json';c.write_text(json.dumps(connection()))
    def forbidden(*a,**kw):raise AssertionError('Network attempted')
    monkeypatch.setattr(cli.subprocess,'run',forbidden)
    report=tmp_path/'dry.json';assert cli.main(['deploy','--connection',str(c),'--bundle',str(z),'--report',str(report)])==0
    data=json.loads(report.read_text());assert data['dry_run'] and not data['remote_execution_performed']
    argv=cli.ssh_argv(connection(),'status',[])
    assert 'StrictHostKeyChecking=yes' in argv and 'BatchMode=yes' in argv
    remote=shlex.split(argv[-1]);assert remote[:2]==['/usr/bin/python3','-c'] and remote[3]=='status'
    assert remote[2]==Path(t.__file__).read_text()


@pytest.mark.parametrize('field,value',[('host','x;touch /tmp/x'),('user','-oProxyCommand=x'),('port',0),('remote_base','/'),('remote_base','/workspace/../etc'),('remote_python','python3;true')])
def test_connection_refuses_shell_or_scope_ambiguity(field,value):
    c=connection();c[field]=value
    with pytest.raises(ValueError):cli.connection(c)


def test_unmanaged_nonempty_remote_base_refused(tmp_path):
    b=tmp_path/'remote';b.mkdir();(b/'existing.txt').write_text('keep')
    with pytest.raises(ValueError,match='unmanaged'):t.layout(b)
    assert (b/'existing.txt').read_text()=='keep'


def test_source_bundle_excludes_prior_results_caches_and_credentials(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    for directory in cli.RELEASE_TOP:(root/directory).mkdir()
    (root/'src/module.py').write_text('pass')
    for name in cli.RELEASE_FILES:(root/name).write_text('fixture')
    for directory in ('results','.ssh','src/__pycache__'):
        p=root/directory;p.mkdir(parents=True,exist_ok=True);(p/'secret.txt').write_text('must-not-upload')
    out=tmp_path/'source.zip';m=cli.source_bundle(root,out)
    assert 'src/module.py' in m['files'] and not any('secret' in x for x in m['files'])


def test_bundle_bytes_are_deterministic_for_retry_and_deduplication(tmp_path):
    z,m=bundle(tmp_path);root=tmp_path/'source';other=tmp_path/'same.zip'
    t.pack(root,[root/'code.py'],other,kind='release',context={'version':'synthetic-test'})
    assert t.sha(z)==t.sha(other)
