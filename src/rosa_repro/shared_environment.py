"""Pinned Linux/CUDA shared-environment recipe; static compatibility is not GPU readiness."""
import hashlib,json,math,re,shlex,subprocess,tomllib
from copy import deepcopy
from pathlib import Path
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name
from .execution_plan import digest
from .regression import UPSTREAM_COMMIT,write_json,sha256

VARIANT='groot-vllm-cu128-v1'
PINS={'torch':'2.9.0+cu128','torchvision':'0.24.0+cu128','torchaudio':'2.9.0+cu128','transformers':'4.57.3',
      'numpy':'1.26.4','opencv-python-headless':'4.11.0.86','vllm':'0.11.2','ray':'2.59.0','pillow':'12.0.0',
      'pydantic':'2.12.5','huggingface-hub':'0.36.2','flash-attn':'2.8.3','pyarrow':'19.0.1',
      'nvidia-cutlass-dsl':'4.2.1'}
SOURCES={'groot':'a451af4cf723cc58b710a2f6f3a12a8d7f03850a39cc5ba420a18fcc20e2a3d0',
         'vllm_cuda':'e424595c46a0a9b1d126911a89a53e99892f7178cabab6a201750905ec335e7a',
         'vllm_common':'99d516f95c8e250e86086f5ccdfb3d501a1dae7d147cec90e8a3261cb7536c09'}


def sources():
    rows=json.loads(Path(__file__).with_name('shared_environment_sources.json').read_text())
    if set(rows)!=set(SOURCES) or any(hashlib.sha256(v['text'].encode()).hexdigest()!=SOURCES[k] or v['sha256']!=SOURCES[k] for k,v in rows.items()):raise ValueError('Pinned dependency sources changed')
    return rows


def requirements():
    rows=sources();g=tomllib.loads(rows['groot']['text']);env=dict(default_environment(),sys_platform='linux',platform_system='Linux',platform_machine='x86_64',python_version='3.12',python_full_version='3.12.0')
    reqs=list(g['project']['dependencies'])
    for key in ('vllm_cuda','vllm_common'):
        reqs += [x.split('#',1)[0].strip() for x in rows[key]['text'].splitlines() if x.strip() and not x.lstrip().startswith(('#','-r'))]
    return g,[Requirement(x) for x in reqs if x and (Requirement(x).marker is None or Requirement(x).marker.evaluate(env))]


def inspect(pins=None):
    pins={canonicalize_name(k):v for k,v in (pins or PINS).items()};g,reqs=requirements();conflicts=[];checked=[]
    for r in reqs:
        name=canonicalize_name(r.name)
        if name in pins:
            ok=r.specifier.contains(pins[name],prereleases=True);checked.append(dict(requirement=str(r),selected=pins[name],compatible=ok))
            if not ok:conflicts.append(str(r))
    # Check every upstream exact direct pin as well, including constraints added
    # by the other project. Unpinned transitive packages still need a Linux resolve.
    for r in reqs:
        exact=[s.version for s in r.specifier if s.operator=='==']
        if exact:
            for other in reqs:
                if canonicalize_name(r.name)==canonicalize_name(other.name) and not other.specifier.contains(exact[0],prereleases=True):conflicts.append(str(r)+' vs '+str(other))
    return dict(variant=VARIANT,status='conflict' if conflicts else 'direct_constraints_compatible',pins=pins,conflicts=sorted(set(conflicts)),checks=checked,
        python='3.12',platform='Linux x86_64',cuda='12.8',groot_commit=UPSTREAM_COMMIT,
        sources={k:{n:v[n] for n in ('url','sha256')} for k,v in sources().items()},
        transitive_resolution_verified=False,CUDA_ABI_verified=False,models_loaded=False,
        next_gate='fresh Linux environment resolve, pip check, module imports, CUDA and real-model smoke; then acquire new profiles')


def dependency_inputs():
    g,reqs=requirements();flash=g['tool']['uv']['sources']['flash-attn'][0]['url']
    lines=sorted(set(str(r) for r in reqs if canonicalize_name(r.name)!='flash-attn'))
    lines += ['flash-attn @ '+flash,'vllm==0.11.2','ray[serve]==2.59.0','jsonschema==4.25.1','packaging==26.3',*g['build-system']['requires']]
    return '\n'.join(lines)+'\n','\n'.join(k+'=='+v for k,v in sorted(PINS.items()))+'\n'


def recipe(*,repo,environment,uv='/usr/local/bin/uv',locked=True):
    paths=[Path(v) for v in (repo,environment,uv)]
    if any(not p.is_absolute() or '..' in p.parts or not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(p)) for p in paths):raise ValueError('Plain absolute paths required')
    repo,environment,uv=map(str,paths)
    if len(Path(environment).parts)<3 or environment in (repo,'/opt/p-ai-policy','/opt/p-ai-libero/.venv'):raise ValueError('Fresh isolated shared environment required')
    report=inspect()
    if report['status']!='direct_constraints_compatible':raise ValueError('Dependency conflict')
    requirement_text,constraints=dependency_inputs()
    value=dict(kind='shared_environment_recipe',variant=VARIANT,repo=repo,environment=environment,uv=uv,inspection=report,
        requirements=requirement_text,constraints=constraints,groot_pyproject_sha256=SOURCES['groot'],
        scope='full pinned GR00T direct dependencies plus vLLM; no hidden removal of training/export dependencies',
        system_requirements=['Linux x86_64, Python 3.12, CUDA-compatible NVIDIA driver','FFmpeg runtime 4..7 (torchcodec 0.8)','sufficient disk for wheels/builds; target environment does not exist'],
        commands=[[uv,'venv','--python','3.12',environment],
          [uv,'pip','install','--python',environment+'/bin/python','--index-strategy','unsafe-best-match','--extra-index-url','https://download.pytorch.org/whl/cu128',
           'setuptools>=77.0.3,<81','wheel','ninja','torch==2.9.0+cu128','torchvision==0.24.0+cu128','torchaudio==2.9.0+cu128'],
          [uv,'pip','install','--python',environment+'/bin/python','--index-strategy','unsafe-best-match','--extra-index-url','https://download.pytorch.org/whl/cu128',
           '--extra-index-url','https://pypi.nvidia.com','--no-build-isolation','-c','constraints.txt','-r','requirements.in'],
          [uv,'pip','install','--python',environment+'/bin/python','--no-deps',repo],
          [uv,'pip','check','--python',environment+'/bin/python'],
          [uv,'pip','freeze','--python',environment+'/bin/python']],
        executable_after_local_checks=False,transitive_lock_available=False)
    value['locked']=locked
    if locked:
        from .shared_lock import load,UV_VERSION
        lock=load();value.update(transitive_lock_available=True,dependency_lock=lock,dependency_graph_resolved=True,uv_version=UV_VERSION)
        common=[uv,'pip','install','--python',environment+'/bin/python','--index-strategy','unsafe-best-match','--require-hashes']
        value['commands'][1]=common+['-r','bootstrap.lock']
        value['commands'][2]=common+['--no-build-isolation','-r','requirements.lock']
        value['commands'][3]=[uv,'pip','install','--python',environment+'/bin/python','--no-deps','--no-build-isolation',repo]
    value['id']=digest(value);return value


def export(value,output):
    if value!=recipe(repo=value['repo'],environment=value['environment'],uv=value['uv'],locked=value['locked']):raise ValueError('Recipe changed')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write_json(out/'recipe.json',value)
    (out/'requirements.in').write_text(value['requirements']);(out/'constraints.txt').write_text(value['constraints'])
    if value['locked']:
        write_json(out/'dependency-lock.json',value['dependency_lock'])
        for name,content in value['dependency_lock']['locks'].items():(out/name).write_text(content)
    # Shell recipe is explicit opt-in on the Pod. Never touch existing policy env.
    q=shlex.quote;lines=['#!/usr/bin/env bash','set -euo pipefail','cd -- "$(dirname -- "$0")"',
        'test "$(uname -s)" = Linux','test "$(uname -m)" = x86_64',f'test ! -e {q(value["environment"])}',
        f'test "$(git -C {q(value["repo"])} rev-parse HEAD)" = {q(UPSTREAM_COMMIT)}',
        f'test -z "$(git -C {q(value["repo"])} status --porcelain --untracked-files=no)"',
        f'test "$(sha256sum {q(value["repo"]+"/pyproject.toml")} | cut -d " " -f 1)" = {q(SOURCES["groot"])}']
    if value['locked']:
        lines.append('test "$('+q(value['uv'])+' --version | cut -d " " -f 2)" = '+q(value['uv_version']))
        for name,content in value['dependency_lock']['locks'].items():
            lines.append('test "$(sha256sum '+q(name)+' | cut -d " " -f 1)" = '+q(hashlib.sha256(content.encode()).hexdigest()))
    for i,command in enumerate(value['commands']):lines.append(shlex.join(command)+(' > installed.freeze.txt' if i==len(value['commands'])-1 else ''))
    (out/'build-on-pod.sh').write_text('\n'.join(lines)+'\n')
    return dict(status='prepared',directory=str(out),models_started=False)


def memory_policy(*,s1_fraction,vlm_fractions,headroom_fraction):
    if set(vlm_fractions)!={'s2','safety','monitor'}:raise ValueError('Explicit budget for every VLM')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<v<1 for v in [s1_fraction,headroom_fraction,*vlm_fractions.values()]):raise ValueError('Finite nonzero fractions required')
    if any(v>.95 for v in vlm_fractions.values()) or sum(vlm_fractions.values())+s1_fraction+headroom_fraction>1+1e-12:raise ValueError('Declared residency budgets exceed one GPU')
    value=dict(kind='shared_residency_budget',s1_fraction=s1_fraction,vlm_fractions=deepcopy(vlm_fractions),headroom_fraction=headroom_fraction,
        total_fraction=s1_fraction+headroom_fraction+sum(vlm_fractions.values()),measured_fit=False,
        semantics='VLM fractions are engine reservations; S1/headroom are planning allowances, not allocator caps or measured peaks')
    value['id']=digest(value);return value


def bind(configs,policy,*,python):
    from .profile_protocol import validate_probe_config
    from .vlm_contract import runtime_fingerprint
    if set(configs)!={'s1','s2','safety','monitor'}:raise ValueError('All four probe configs required')
    if policy!=memory_policy(**{k:policy[k] for k in ('s1_fraction','vlm_fractions','headroom_fraction')}):raise ValueError('Memory policy changed')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',python) or '..' in Path(python).parts:raise ValueError('Absolute common Python required')
    out=deepcopy(configs)
    for role,c in out.items():
        validate_probe_config(c,ready=False)
        if c['role']!=role or c['mode']!='gpu':raise ValueError('GPU role binding required')
        c['binding']['runtime_env']={'py_executable':python}
        if role!='s1':
            if c['binding']['kwargs']['config']['engine']!='vllm':raise ValueError('Shared VLM plan requires vLLM configs')
            c['binding']['kwargs']['runtime_variant']=VARIANT
            c['binding']['kwargs']['config']['gpu_memory_utilization']=policy['vlm_fractions'][role]
            c['runtime_fingerprint']=runtime_fingerprint(c['binding']['kwargs']['config'],VARIANT)
        else:c['runtime_fingerprint']='shared-s1-'+digest(dict(previous=c['runtime_fingerprint'],variant=VARIANT,policy=policy,python=python))
        validate_probe_config(c,ready=False)
    return dict(kind='shared_probe_bindings',configs=out,memory_policy=policy,requires_new_profiles=True,requires_new_R06_for_S1=True,GPU_ready=False)


def preflight():
    """Inspect installed metadata first; no model load and no installation."""
    import platform,sys,importlib,importlib.metadata as metadata
    from .shared_lock import installed_versions
    observed,errors=installed_versions(metadata.version)
    for name,want in PINS.items():
        try:
            v=metadata.version(name);observed[name]=v
            if not Requirement(name+'=='+want).specifier.contains(v,prereleases=True):errors.append(name+': version differs')
        except metadata.PackageNotFoundError:errors.append(name+': missing')
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,12):errors.append('target platform differs')
    for dist in metadata.distributions():
        for raw in dist.requires or []:
            req=Requirement(raw)
            if req.marker and not req.marker.evaluate():continue
            try:v=metadata.version(req.name)
            except metadata.PackageNotFoundError:errors.append(str(req)+': missing');continue
            if not req.specifier.contains(v,prereleases=True):errors.append(str(req)+': unsatisfied')
    imports=[];cuda=False
    if not errors:
        for module in ('gr00t.policy.gr00t_policy','vllm','transformers','flash_attn','torchcodec','pyarrow','ray.serve',
                       'vllm.model_executor.models.qwen2_5_vl'):
            try:importlib.import_module(module);imports.append(module)
            except Exception as exc:errors.append(module+': '+type(exc).__name__)
        if 'pyarrow' in imports:
            # Metadata alone misses NumPy ABI incompatibilities in binary wheels.
            # Exercise the Arrow/NumPy conversion used by Ray's serializers.
            try:
                import numpy as np
                import pyarrow as pa
                values=np.array([1.,2.],dtype=np.float64)
                if not np.array_equal(pa.array(values).to_numpy(),values):raise ValueError('Arrow roundtrip differs')
                from ray._private.serialization import SerializationContext
                if SerializationContext(None).serialize(values).total_bytes<=0:raise ValueError('Empty Ray serialization')
            except Exception as exc:errors.append('pyarrow_numpy_roundtrip: '+type(exc).__name__)
        import torch
        cuda=torch.cuda.is_available() and torch.version.cuda=='12.8'
        if not cuda:errors.append('CUDA 12.8 unavailable')
    return dict(status='passed' if not errors else 'blocked',variant=VARIANT,versions=observed,imports=imports,cuda_available=cuda,
        errors=sorted(set(errors)),models_loaded=False,joint_residency_verified=False)
