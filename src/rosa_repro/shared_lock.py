"""Linux-only dependency resolution with a reusable hashed lock, not GPU validation.

May run as a standalone file in a minimal resolver venv; repository imports are
only needed to prepare/validate the pinned project's input files.
"""
import hashlib,json,os,platform,re,subprocess,sys,time
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

UV_VERSION='0.13.6'
INDEXES=['https://pypi.org/simple','https://download.pytorch.org/whl/cu128','https://pypi.nvidia.com']
BUILD_INPUT='setuptools>=77.0.3,<81\nwheel\npip\nninja\ntorch==2.9.0+cu128\ntorchvision==0.24.0+cu128\ntorchaudio==2.9.0+cu128\n'

def hash_text(s):return hashlib.sha256(s.encode()).hexdigest()
def identity(v):return hash_text(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False))
def dump(path,v):Path(path).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def inputs():
    from .shared_environment import dependency_inputs,VARIANT,SOURCES
    requirements,constraints=dependency_inputs()
    files={'requirements.in':requirements,'constraints.txt':constraints,'bootstrap.in':BUILD_INPUT}
    return dict(kind='shared_lock_inputs',variant=VARIANT,upstream_sources=SOURCES,files=files,
                hashes={k:hash_text(v) for k,v in files.items()})

def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);value=inputs();dump(out/'inputs.json',value)
    for name,content in value['files'].items():(out/name).write_text(content)
    return value

def entries(content):
    """Reject unconstrained packages, missing hashes and unexpected pip options."""
    rows={};index=[]
    for line in content.replace('\\\n',' ').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        if line.startswith(('--index-url ','--extra-index-url ')):
            index.append(line.split()[1]);continue
        parts=line.split(' --hash=sha256:');req=Requirement(parts[0].strip());name=canonicalize_name(req.name)
        hashes=[p.strip() for p in parts[1:]]
        if not hashes or any(not re.fullmatch('[a-f0-9]{64}',h) for h in hashes):raise ValueError('Every locked requirement needs SHA256 hashes')
        pins=list(req.specifier)
        if req.marker or (not req.url and (len(pins)!=1 or pins[0].operator!='==' or '*' in pins[0].version)):raise ValueError('Exact target requirement required')
        if req.url and not req.url.startswith('https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/'):raise ValueError('Unexpected direct artifact')
        if name in rows:
            prior=Requirement(rows[name]['requirement'])
            if prior.specifier!=req.specifier or prior.url!=req.url or rows[name]['hashes']!=sorted(hashes):raise ValueError('Conflicting duplicate locked package')
            req.extras |= prior.extras
        rows[name]=dict(requirement=str(req),hashes=sorted(hashes))
    if index!=INDEXES:raise ValueError('Lock indexes differ')
    return rows

def validate(value,*,expected=None):
    expected=inputs() if expected is None else expected
    payload={k:v for k,v in value.items() if k!='id'}
    if value['id']!=identity(payload) or value['inputs']!=expected:raise ValueError('Lock input or identity differs')
    if value['target']!={'system':'Linux','machine':'x86_64','python':'3.12'} or value['uv_version']!=UV_VERSION:raise ValueError('Resolver target/version differs')
    if value['status']!='resolved' or not value['locked_recheck_passed']:raise ValueError('Unresolved dependencies')
    if [c['name'] for c in value['commands']]!=['uv-version','resolve','bootstrap','locked-recheck','offline-recheck'] or any((c['exit_code']!=0 and c['name']!='offline-recheck') or not re.fullmatch('[a-f0-9]{64}',c['log_sha256']) for c in value['commands']):raise ValueError('Missing or failed resolver evidence')
    if value['offline_recheck_passed']!=(value['commands'][-1]['exit_code']==0):raise ValueError('Offline result differs')
    for k in ('all_packages_installed','CUDA_ABI_verified','models_loaded'):
        if value[k] is not False:raise ValueError('Dependency resolution cannot certify runtime')
    locks=value['locks'];rows={k:entries(v) for k,v in locks.items()}
    if set(rows)!={'requirements.lock','bootstrap.lock'}:raise ValueError('Both dependency and build bootstrap locks required')
    for k,v in rows['bootstrap.lock'].items():
        main=rows['requirements.lock'][k];a=Requirement(v['requirement']);b=Requirement(main['requirement'])
        if a.specifier!=b.specifier or a.url!=b.url or not a.extras.issubset(b.extras) or v['hashes']!=main['hashes']:raise ValueError('Bootstrap resolution differs from runtime lock')
    for name,content in expected['files'].items():
        if expected['hashes'][name]!=hash_text(content):raise ValueError('Input hash differs')
        for line in content.splitlines():
            r=Requirement(line);name=canonicalize_name(r.name);selected=Requirement(rows['requirements.lock'][name]['requirement'])
            if r.url and selected.url!=r.url:raise ValueError('Direct wheel differs')
            if not selected.url and not r.specifier.contains(next(iter(selected.specifier)).version,prereleases=True):raise ValueError('Direct constraint differs')
            if not r.extras.issubset(selected.extras):raise ValueError('Required extras removed')
    if value['package_count']!=len(rows['requirements.lock']):raise ValueError('Package count differs')
    return value

def load():return validate(json.loads(Path(__file__).with_name('shared_environment_lock.json').read_text()))

def installed_versions(version):
    from importlib.metadata import PackageNotFoundError
    from .shared_environment import PINS
    observed={};errors=[]
    for name,row in entries(load()['locks']['requirements.lock']).items():
        req=Requirement(row['requirement'])
        wanted='=='+PINS[name] if req.url else str(req.specifier)
        try:
            actual=version(name);observed[name]=actual
            if not Requirement(name+wanted).specifier.contains(actual,prereleases=True):errors.append(name+': locked version differs')
        except PackageNotFoundError:errors.append(name+': locked dependency missing')
    return observed,errors

def resolve(input_dir,out,uv,prefer_lock=None):
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,12):raise ValueError('Run inside Linux x86_64 Python 3.12; cross-platform sdist metadata is insufficient')
    source=Path(input_dir).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    expected=json.loads((source/'inputs.json').read_text())
    for k,v in expected['files'].items():
        if (source/k).read_text()!=v or hash_text(v)!=expected['hashes'][k]:raise ValueError('Prepared input changed')
        (out/k).write_text(v)
    preferred=None
    if prefer_lock is not None:
        preferred=Path(prefer_lock).read_text();entries(preferred)
        # uv preserves already-locked versions unless new constraints conflict.
        # This prevents an ABI repair from upgrading unrelated dependencies.
        (out/'requirements.lock').write_text(preferred)
    env=dict(os.environ,UV_NO_PROGRESS='1',UV_HTTP_TIMEOUT='60',DS_BUILD_OPS='0')
    commands=[]
    def run(name,args,required=True):
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as stream:
            result=subprocess.run(args,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        row=dict(name=name,argv=args,exit_code=result.returncode,elapsed_s=time.monotonic()-start,log_sha256=hash_text((out/(name+'.log')).read_text()))
        commands.append(row);dump(out/'commands.json',commands)
        if result.returncode and required:raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))
        return (out/(name+'.log')).read_text()
    version=run('uv-version',[uv,'--version']).split()[1]
    if version!=UV_VERSION:raise ValueError('Pinned uv '+UV_VERSION+' required')
    common=[uv,'pip','compile','--python',sys.executable,'--python-version','3.12','--index-strategy','unsafe-best-match','--index-url',INDEXES[0],
        '--extra-index-url',INDEXES[1],'--extra-index-url',INDEXES[2],'--no-build-isolation','--generate-hashes','--emit-index-url','--no-strip-extras','--no-header','--no-annotate','--quiet']
    run('resolve',common+[str(out/'requirements.in'),'-c',str(out/'constraints.txt'),'-o',str(out/'requirements.lock')])
    run('bootstrap',common+[str(out/'bootstrap.in'),'-c',str(out/'requirements.lock'),'-o',str(out/'bootstrap.lock')])
    run('locked-recheck',common+[str(out/'requirements.in'),'-c',str(out/'requirements.lock'),'-o',str(out/'rechecked.lock')])
    run('offline-recheck',common+['--offline',str(out/'requirements.in'),'-c',str(out/'requirements.lock'),'-o',str(out/'offline.lock')],required=False)
    locks={k:(out/k).read_text() for k in ('requirements.lock','bootstrap.lock')}
    if entries((out/'rechecked.lock').read_text())!=entries(locks['requirements.lock']):raise ValueError('Locked dependency closure differs')
    offline_ok=commands[-1]['exit_code']==0
    if offline_ok and entries((out/'offline.lock').read_text())!=entries(locks['requirements.lock']):raise ValueError('Offline dependency closure differs')
    value=dict(kind='shared_environment_lock',status='resolved',target=dict(system='Linux',machine='x86_64',python='3.12'),
        uv_version=version,inputs=expected,locks=locks,package_count=len(entries(locks['requirements.lock'])),commands=commands,
        platform=dict(python=sys.version,libc=list(platform.libc_ver()),platform=platform.platform()),
        locked_recheck_passed=True,offline_recheck_passed=offline_ok,all_packages_installed=False,CUDA_ABI_verified=False,models_loaded=False)
    if preferred is not None:value['preferred_lock_sha256']=hash_text(preferred)
    value['id']=identity(value);validate(value,expected=expected);dump(out/'shared_environment_lock.json',value);return value

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--uv',required=True);p.add_argument('--prefer-lock');a=p.parse_args()
    result=resolve(a.inputs,a.output,a.uv,a.prefer_lock);print(json.dumps({k:result[k] for k in ('id','status','package_count','offline_recheck_passed')}))
