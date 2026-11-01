import asyncio,json
import pytest
from rosa_repro import staged_capture as module
from rosa_repro.staged_capture import create


def capture(tmp_path,maximum=16*1024**2):
    local=tmp_path/'local';local.mkdir(exist_ok=True)
    return create(local,tmp_path/'persistent'/'run',maximum_bytes=maximum,require_local=False)


def populate(c):
    c.output.mkdir();(c.output/'data').write_bytes(b'original observation bytes')


def test_publication_preserves_bytes_and_removes_only_verified_source(tmp_path):
    c=capture(tmp_path);populate(c);receipt=c.publish()
    assert c.output.exists() and receipt['copied_verified']
    assert not receipt['deadline_timestamps_modified'] and not receipt['request_cohort_modified']
    assert (c.destination/'data').read_bytes()==b'original observation bytes'
    c.remove_verified_source();assert not c.directory.exists()
    assert json.loads((c.destination/'capture_source_cleanup.json').read_text())['source_removed']


def test_copy_failure_retains_local_original(tmp_path,monkeypatch):
    c=capture(tmp_path);populate(c)
    def fail(*args,**kwargs):raise OSError('injected persistent storage failure')
    monkeypatch.setattr(module.shutil,'copytree',fail)
    with pytest.raises(OSError):c.publish()
    assert (c.output/'data').read_bytes()==b'original observation bytes'
    assert not c.destination.exists()


def test_corrupt_copy_is_not_published_as_verified(tmp_path,monkeypatch):
    c=capture(tmp_path);populate(c);original=module.shutil.copytree
    def corrupt(src,dst):
        original(src,dst);(dst/'data').write_bytes(b'corrupted')
    monkeypatch.setattr(module.shutil,'copytree',corrupt)
    with pytest.raises(ValueError,match='changed during'):c.publish()
    assert c.output.exists() and not (c.destination/'capture_transfer.json').exists()


def test_persistent_tampering_prevents_source_removal(tmp_path):
    c=capture(tmp_path);populate(c);c.publish();(c.destination/'data').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='local source retained'):c.remove_verified_source()
    assert c.output.exists()


def test_cannot_overwrite_destination_or_publish_external_links(tmp_path):
    c=capture(tmp_path);populate(c);c.destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):c.publish()
    c.destination.rmdir();(c.output/'link').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='symlink'):c.publish()
    assert c.output.exists()


def test_budget_and_explicit_local_filesystem_required(tmp_path,monkeypatch):
    c=capture(tmp_path,maximum=2);populate(c)
    with pytest.raises(ValueError,match='byte budget'):c.publish()
    info=dict(filesystem_type='fuse',available_bytes=2**40)
    monkeypatch.setattr(module,'storage',lambda p:info)
    with pytest.raises(ValueError,match='local capture filesystem'):create(tmp_path,tmp_path/'fresh',maximum_bytes=10)
    info.update(filesystem_type='overlay',available_bytes=0)
    with pytest.raises(ValueError,match='headroom'):create(tmp_path,tmp_path/'fresh',maximum_bytes=10)


def test_output_root_cannot_redirect_publication_outside_capture(tmp_path):
    c=capture(tmp_path);external=tmp_path/'external';external.mkdir();(external/'secret').write_text('private')
    c.output.symlink_to(external,target_is_directory=True)
    with pytest.raises(ValueError,match='owned directory'):c.publish()
    assert not c.destination.exists() and (external/'secret').exists()


def test_closed_loop_reaudits_identically_after_publication(tmp_path):
    from test_closed_loop import suite,Sim
    from rosa_repro.local_fleet import LocalFleetTransport
    from rosa_repro.closed_loop_runtime import run_closed_loop,reaudit_closed
    s=suite(tmp_path);c=capture(tmp_path);sims={r['robot_id']:Sim() for x in s['plan']['classes'] for r in x['robots']}
    async def go():
        t=LocalFleetTransport(s['plan'])
        try:return await run_closed_loop(s,tmp_path,t,sims,c.output/'raw')
        finally:await t.drain()
    assert asyncio.run(go())['status']=='complete'
    before=reaudit_closed(c.output/'raw');c.publish()
    assert reaudit_closed(c.destination/'raw')==before
    c.remove_verified_source()
