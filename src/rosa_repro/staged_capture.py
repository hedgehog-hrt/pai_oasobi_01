"""Bounded local capture followed by verified publication to persistent storage.

This changes the recording filesystem, never the request clock or its cohort.
The source is retained on every copy/verification failure.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .regression import sha256,write_json
from .execution_plan import digest


def storage(path):
    path=Path(path).resolve(strict=True);stat=os.statvfs(path);kind='not_observed'
    mounts=Path('/proc/self/mountinfo')
    if mounts.is_file():
        matches=[]
        for line in mounts.read_text().splitlines():
            left,right=line.split(' - ',1);mount=Path(left.split()[4].replace('\\040',' '))
            if path==mount or mount in path.parents:matches.append((len(mount.parts),right.split()[0]))
        if matches:kind=max(matches)[1]
    return dict(path=str(path),filesystem_type=kind,available_bytes=stat.f_bavail*stat.f_frsize,
        total_bytes=stat.f_blocks*stat.f_frsize)


def manifest(directory,maximum):
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():raise ValueError('Capture root is not an owned directory')
    files={};total=0
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():raise ValueError('Capture contains a symlink')
        if not p.is_file():continue
        if p.stat().st_nlink!=1:raise ValueError('Capture contains a hardlink')
        size=p.stat().st_size;total+=size
        if total>maximum:raise ValueError('Capture publication byte budget exceeded')
        files[str(p.relative_to(directory))]=dict(size=size,sha256=sha256(p))
    if not files:raise ValueError('Capture is empty')
    return dict(files=files,total_bytes=total,sha256=digest(files))


@dataclass
class Capture:
    directory: Path
    destination: Path
    owner: dict

    @property
    def output(self):return self.directory/'run'

    def check_owner(self):
        if self.directory.is_symlink() or self.directory.resolve()!=self.directory:
            raise ValueError('Capture ownership path changed')
        if json.loads((self.directory/'owner.json').read_text())!=self.owner:
            raise ValueError('Capture ownership marker changed')

    def publish(self):
        self.check_owner()
        if self.destination.exists():raise FileExistsError(self.destination)
        before=manifest(self.output,self.owner['maximum_bytes'])
        if {'capture_transfer.json','capture_source_cleanup.json'}&set(before['files']):
            raise ValueError('Capture contains a reserved publication record')
        self.destination.parent.mkdir(parents=True,exist_ok=True)
        if self.destination.parent.resolve()!=self.destination.parent:raise ValueError('Destination parent changed')
        begun=time.monotonic_ns()
        # copytree creates the destination exclusively; a partial copy is never
        # marked verified, and is retained beside its intact local source.
        shutil.copytree(self.output,self.destination)
        copied=manifest(self.destination,self.owner['maximum_bytes'])
        after=manifest(self.output,self.owner['maximum_bytes'])
        if before!=copied or before!=after:raise ValueError('Capture changed during publication')
        receipt=dict(protocol='verified_local_capture_v1',source=str(self.output),destination=str(self.destination),
            owner=self.owner,manifest=before,copied_verified=True,copy_started_ns=begun,copy_finished_ns=time.monotonic_ns(),
            deadline_timestamps_modified=False,request_cohort_modified=False)
        write_json(self.destination/'capture_transfer.json',receipt)
        return receipt

    def remove_verified_source(self):
        self.check_owner()
        receipt=json.loads((self.destination/'capture_transfer.json').read_text())
        if receipt['owner']!=self.owner or not receipt['copied_verified']:raise ValueError('Publication receipt differs')
        expected=receipt['manifest']
        if manifest(self.output,self.owner['maximum_bytes'])!=expected:raise ValueError('Source changed after publication')
        for name,row in expected['files'].items():
            p=self.destination/name
            if p.is_symlink() or p.stat().st_size!=row['size'] or sha256(p)!=row['sha256']:
                raise ValueError('Persistent copy changed; local source retained')
        shutil.rmtree(self.directory)
        write_json(self.destination/'capture_source_cleanup.json',dict(source_removed=True,persistent_copy_verified=True))


def create(root,destination,*,maximum_bytes,require_local=True):
    root=Path(root).resolve(strict=True);destination=Path(destination).absolute()
    if not root.is_dir() or destination.exists():raise ValueError('Existing capture root and fresh destination required')
    if type(maximum_bytes) is not int or not 0<maximum_bytes<=20*1024**3:raise ValueError('Bounded capture size required')
    info=storage(root)
    if require_local and info['filesystem_type'] not in ('overlay','tmpfs','ext4','xfs','btrfs'):
        raise ValueError('Explicit local capture filesystem required')
    if info['available_bytes']<maximum_bytes+512*1024**2:raise ValueError('Insufficient local capture headroom')
    directory=Path(tempfile.mkdtemp(prefix='rosa-capture-',dir=root)).resolve()
    owner=dict(protocol='local_capture_owner_v1',nonce=uuid.uuid4().hex,directory=str(directory),destination=str(destination),
        maximum_bytes=maximum_bytes,storage=info,created_ns=time.monotonic_ns(),requires_persistent_copy_before_removal=True)
    write_json(directory/'owner.json',owner)
    return Capture(directory,destination,owner)
