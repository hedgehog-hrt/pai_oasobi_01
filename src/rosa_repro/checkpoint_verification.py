"""Streaming, offline verification of the pinned GR00T inference checkpoint.
Never downloads weights or deserializes pickle/model tensors.
"""
import hashlib
from pathlib import Path
from .contracts import load_json,validate,SHA256
from .execution_plan import digest
from .regression import MODEL_REVISION,T01_CONFIG_HASHES


def groot_pin():
    p=load_json(Path(__file__).with_name('groot_checkpoint_pin.json'))
    if (p['model_id'],p['revision'],p['subdirectory'])!=('nvidia/GR00T-N1.7-LIBERO',MODEL_REVISION,'libero_object'):
        raise ValueError('Checkpoint pin identity differs from supported baseline')
    for name,h in T01_CONFIG_HASHES.items():
        if p['files'][name]['sha256']!=h:raise ValueError('Pinned config/index differs from archived T01')
    return p


def _verify_directory(directory,pin):
    directory=Path(directory);checked={};total=0
    weights={n for n,r in pin['files'].items() if r['kind']=='weight'}
    if not weights:raise ValueError('Pin contains no model weights')
    # A loader could prefer a different unsharded/legacy checkpoint if present.
    alternatives={p.name for p in directory.iterdir() if p.is_file() and
                  (p.name.endswith('.safetensors') or p.name.startswith(('pytorch_model','adapter_model')) or p.name=='adapter_config.json')}
    if alternatives-weights:raise ValueError('Unverified alternative model files: '+str(sorted(alternatives-weights)))
    for name,row in pin['files'].items():
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Invalid pinned filename')
        validate(row['sha256'],SHA256)
        path=directory/name
        if not path.is_file():raise ValueError('Missing required inference file: '+name)
        before=path.stat()
        if before.st_size!=row['size']:raise ValueError('Wrong file size: '+name)
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        after=path.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('File changed during verification: '+name)
        if h.hexdigest()!=row['sha256']:raise ValueError('Inference file SHA256 mismatch: '+name)
        checked[name]=dict(sha256=h.hexdigest(),size=before.st_size,kind=row['kind']);total+=before.st_size
    index=load_json(directory/'model.safetensors.index.json');mapping=index['weight_map']
    if not isinstance(mapping,dict) or len(mapping)!=pin['index_tensor_count'] or set(mapping.values())!=weights:
        raise ValueError('Index/shard coverage differs from pin')
    if index['metadata']['total_size']!=pin['index_total_tensor_bytes']:raise ValueError('Index tensor size differs')
    return dict(status='passed',model_id=pin['model_id'],revision=pin['revision'],subdirectory=pin['subdirectory'],
        scope=pin['scope'],pin_sha256=digest(pin),files=checked,bytes_checked=total,weight_files_checked=len(weights),
        full_weight_hashes_verified=True,index_shard_coverage_verified=True,model_loaded=False,gpu_execution=False,
        external_runtime_dependencies_verified=False,training_resume_artifacts_verified=False)


def verify_groot_checkpoint(directory):return _verify_directory(directory,groot_pin())
