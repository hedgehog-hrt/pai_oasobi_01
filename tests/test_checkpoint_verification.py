from copy import deepcopy
import hashlib,json
from pathlib import Path
import pytest
from rosa_repro.checkpoint_verification import groot_pin,_verify_directory,verify_groot_checkpoint


def fixture(tmp_path):
    index={'metadata':{'total_size':9},'weight_map':{'tensor.a':'model-1.safetensors','tensor.b':'model-2.safetensors'}}
    data={'model-1.safetensors':b'first','model-2.safetensors':b'last',
          'model.safetensors.index.json':json.dumps(index).encode(),'config.json':b'{}','statistics.json':b'{}'}
    for n,b in data.items():(tmp_path/n).write_bytes(b)
    pin=dict(model_id='synthetic-test',revision='a'*40,subdirectory='test',scope='CPU fixture only',index_tensor_count=2,index_total_tensor_bytes=9,
        files={n:dict(size=len(b),sha256=hashlib.sha256(b).hexdigest(),kind='weight' if n.endswith('.safetensors') else 'inference_config') for n,b in data.items()})
    return pin


def test_trusted_pin_matches_baseline_and_covers_both_weight_shards():
    p=groot_pin();assert p['weights_total_bytes']==6910499416 and p['index_tensor_count']==1031
    assert len([r for r in p['files'].values() if r['kind']=='weight'])==2
    assert len(p['files'])==7


def test_all_bytes_and_index_are_verified_without_tensor_deserialization(tmp_path):
    p=fixture(tmp_path);r=_verify_directory(tmp_path,p)
    assert r['weight_files_checked']==2 and r['full_weight_hashes_verified'] and not r['model_loaded']
    assert not r['external_runtime_dependencies_verified']


@pytest.mark.parametrize('bad',['missing','size','hash','index','extra','adapter','traversal'])
def test_bad_or_ambiguous_snapshot_is_rejected(tmp_path,bad):
    p=fixture(tmp_path)
    if bad=='missing':(tmp_path/'model-2.safetensors').unlink()
    if bad=='size':(tmp_path/'model-1.safetensors').write_bytes(b'x')
    if bad=='hash':(tmp_path/'model-1.safetensors').write_bytes(b'FIRST')
    if bad=='index':p['index_tensor_count']=3
    if bad=='extra':(tmp_path/'model.safetensors').write_bytes(b'alternative')
    if bad=='adapter':(tmp_path/'adapter_config.json').write_bytes(b'{}')
    if bad=='traversal':p['files']['../elsewhere']=p['files'].pop('config.json')
    with pytest.raises(ValueError):_verify_directory(tmp_path,p)


def test_HF_cache_symlinks_are_checked_by_content(tmp_path):
    p=fixture(tmp_path);target=tmp_path/'blob';(tmp_path/'model-1.safetensors').rename(target)
    (tmp_path/'model-1.safetensors').symlink_to(target)
    assert _verify_directory(tmp_path,p)['full_weight_hashes_verified']
    target.write_bytes(b'FIRST')
    with pytest.raises(ValueError,match='SHA256'):_verify_directory(tmp_path,p)


def test_real_pin_refuses_small_fixture(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError):verify_groot_checkpoint(tmp_path)
