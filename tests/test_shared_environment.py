from copy import deepcopy
import subprocess
import pytest
from rosa_repro.shared_environment import inspect,recipe,export,memory_policy,bind,VARIANT,PINS
from rosa_repro.profile_protocol import template

def test_pinned_upstream_constraints_detect_old_transformers_conflict():
 r=inspect();assert r['status']=='direct_constraints_compatible' and not r['CUDA_ABI_verified'] and not r['transitive_resolution_verified']
 bad=dict(PINS,transformers='4.57.6');assert inspect(bad)['status']=='conflict'
 assert inspect(dict(PINS,torch='2.8.0'))['status']=='conflict'

def test_recipe_is_isolated_and_shell_syntax_valid(tmp_path):
 r=recipe(repo='/workspace/Isaac-GR00T',environment='/opt/p-ai-shared-test');export(r,tmp_path/'recipe')
 subprocess.run(['bash','-n',str(tmp_path/'recipe/build-on-pod.sh')],check=True)
 assert 'transformers==4.57.3' in r['constraints'] and any('--no-deps' in c for c in r['commands'])
 assert r['transitive_lock_available'] and not r['executable_after_local_checks']
 for path in ('/opt/p-ai-policy','/opt/p-ai-libero/.venv','/opt/x/../existing','/opt/x;bad'):
  with pytest.raises(ValueError):recipe(repo='/workspace/Isaac-GR00T',environment=path)

def test_memory_binding_changes_profiles_and_preserves_inputs():
 c={r:template(r,'groot' if r=='s1' else 'vllm') for r in ('s1','s2','safety','monitor')};old=deepcopy(c)
 p=memory_policy(s1_fraction=.3,vlm_fractions={'s2':.15,'safety':.1,'monitor':.15},headroom_fraction=.3)
 result=bind(c,p,python='/opt/p-ai-shared/bin/python');assert c==old and result['requires_new_profiles'] and result['requires_new_R06_for_S1']
 for role,v in result['configs'].items():
  assert v['binding']['runtime_env']['py_executable']=='/opt/p-ai-shared/bin/python' and v['runtime_fingerprint']!=old[role]['runtime_fingerprint']
  if role!='s1':assert v['binding']['kwargs']['runtime_variant']==VARIANT
 with pytest.raises(ValueError):memory_policy(s1_fraction=.3,vlm_fractions={'s2':.8,'safety':.8,'monitor':.8},headroom_fraction=.1)

def test_shared_variant_is_explicit_and_original_gate_is_unchanged(monkeypatch):
 from rosa_repro import vlm_backends as b
 versions={'torch':'2.9.0+cu128','transformers':'4.57.3','Pillow':'12.0.0','vllm':'0.11.2'}
 monkeypatch.setattr(b,'version',lambda k:versions[k])
 assert b.check_versions('vllm',VARIANT)['transformers']=='4.57.3'
 with pytest.raises(ValueError):b.check_versions('vllm')
 with pytest.raises(ValueError):b.check_versions('vllm','unreviewed')

def test_environment_switch_alone_changes_identity():
 from rosa_repro.vlm_contract import default_config,runtime_fingerprint
 from rosa_repro.profile_protocol import validate_probe_config
 cfg=default_config('s2','vllm')
 assert runtime_fingerprint(cfg)!=runtime_fingerprint(cfg,VARIANT)
 c=template('s2','vllm');c['binding']['kwargs']['runtime_variant']=VARIANT
 with pytest.raises(ValueError):validate_probe_config(c,ready=False)
 c['runtime_fingerprint']=runtime_fingerprint(cfg,VARIANT);validate_probe_config(c,ready=False)

def test_preflight_rejects_binary_import_failure_despite_valid_metadata(monkeypatch):
 import importlib,importlib.metadata,platform
 import torch
 from rosa_repro.shared_environment import preflight
 monkeypatch.setattr('rosa_repro.shared_lock.installed_versions',lambda version:(dict(PINS),[]))
 monkeypatch.setattr(importlib.metadata,'version',lambda name:'2.8.3+cu12torch2.9cxx11abitrue' if name=='flash-attn' else PINS[name])
 monkeypatch.setattr(importlib.metadata,'distributions',lambda:[])
 monkeypatch.setattr(platform,'system',lambda:'Linux')
 monkeypatch.setattr(platform,'machine',lambda:'x86_64')
 checked=[]
 def load(name):
  checked.append(name)
  if name=='pyarrow':raise ImportError('NumPy ABI incompatible')
 monkeypatch.setattr(importlib,'import_module',load)
 monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
 monkeypatch.setattr(torch.version,'cuda','12.8')
 result=preflight()
 assert result['status']=='blocked' and result['errors']==['pyarrow: ImportError']
 assert 'ray.serve' in checked and not result['models_loaded']
