from copy import deepcopy
from pathlib import Path
import subprocess
import pytest
from rosa_repro.shared_lock import load,validate,identity,entries,prepare,resolve
from rosa_repro.shared_environment import recipe,export,PINS


def test_real_linux_lock_contains_full_dependencies_and_build_bootstrap():
 v=load();r=entries(v['locks']['requirements.lock']);b=entries(v['locks']['bootstrap.lock'])
 assert len(r)>200 and {'deepspeed','tensorrt-cu12','tensorrt-cu12-libs','vllm','flash-attn','ray'}<=r.keys()
 assert {'torch','torchvision','torchaudio','pip','wheel','setuptools','ninja'}<=b.keys()
 assert v['locked_recheck_passed'] and not v['CUDA_ABI_verified'] and not v['all_packages_installed']
 assert PINS['flash-attn']=='2.8.3' # distribution metadata; CUDA ABI remains in the fixed wheel URL
 assert 'cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' in r['flash-attn']['requirement']


@pytest.mark.parametrize('change',[
 lambda v:v.update(target=dict(system='Darwin',machine='arm64',python='3.12')),
 lambda v:v.update(locked_recheck_passed=False),
 lambda v:v.update(CUDA_ABI_verified=True),
 lambda v:v['commands'][1].update(exit_code=1),
 lambda v:v.update(package_count=1),
 lambda v:v['locks'].update(**{'bootstrap.lock':v['locks']['bootstrap.lock'].replace('torch==2.9.0+cu128','torch==2.8.0')}),
 lambda v:v['inputs']['files'].update(**{'constraints.txt':'transformers==4.57.6\n'}),
])
def test_resealed_invalid_lock_is_rejected(change):
 v=deepcopy(load());change(v);v['id']=identity({k:x for k,x in v.items() if k!='id'})
 with pytest.raises((ValueError,KeyError)):validate(v)


def test_changed_bytes_are_rejected_before_export(tmp_path):
 v=recipe(repo='/workspace/Isaac-GR00T',environment='/opt/p-ai-shared-new');v['dependency_lock']['locks']['requirements.lock']+='\n# changed\n'
 with pytest.raises(ValueError):export(v,tmp_path/'out')


def test_export_uses_hash_locked_bootstrap_and_all_dependencies(tmp_path):
 v=recipe(repo='/workspace/Isaac-GR00T',environment='/opt/p-ai-shared-new');export(v,tmp_path/'out')
 assert all('--require-hashes' in c for c in v['commands'][1:3])
 assert '--no-build-isolation' in v['commands'][3]
 script=(tmp_path/'out/build-on-pod.sh').read_text();assert 'sha256sum requirements.lock' in script and '--version' in script
 subprocess.run(['bash','-n',str(tmp_path/'out/build-on-pod.sh')],check=True)
 assert not recipe(repo='/workspace/Isaac-GR00T',environment='/opt/p-ai-shared-new',locked=False)['transitive_lock_available']


def test_resolver_rejects_wrong_platform_without_creating_output(tmp_path,monkeypatch):
 import rosa_repro.shared_lock as lock
 monkeypatch.setattr(lock.platform,'system',lambda:'Darwin')
 with pytest.raises(ValueError,match='Linux'):resolve(tmp_path,tmp_path/'out','uv')
 assert not (tmp_path/'out').exists()


def test_lock_parser_rejects_unhashed_unpinned_or_wrong_index():
 for s in ('numpy>=1','numpy==1 --hash=sha256:bad','--extra-index-url https://example.invalid\nnumpy==1'):
  with pytest.raises(ValueError):entries(s)


def test_installed_check_detects_transitive_drift_and_missing_package():
 from packaging.requirements import Requirement
 from importlib.metadata import PackageNotFoundError
 from rosa_repro.shared_lock import installed_versions
 versions={}
 for name,row in entries(load()['locks']['requirements.lock']).items():
  r=Requirement(row['requirement']);versions[name]=PINS[name] if r.url else next(iter(r.specifier)).version
 assert installed_versions(versions.__getitem__)[1]==[]
 versions['accelerate']='0.0.0'
 def actual(name):
  if name=='deepspeed':raise PackageNotFoundError(name)
  return versions[name]
 errors=installed_versions(actual)[1]
 assert errors==['accelerate: locked version differs','deepspeed: locked dependency missing']
