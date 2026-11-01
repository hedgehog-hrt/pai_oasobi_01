"""Fixed-step, budgeted Pod-resumption workflow. Default check performs no execution.
One profile case per step; immutable attempts, successful skips, explicit failures.
"""
from pathlib import Path
import json,math,os,signal,subprocess,time
from .execution_plan import digest,runtime_source_manifest
from .profile_adoption import read
from .profile_protocol import validate_probe_config,validate_suite,load_corpus
from .profile_runner import verify_evidence,reaudit
from .regression import sha256,write_json
from .experiments import ExperimentStore

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=('postpod_models.py','runtime_evidence.py','resume_workflow.py')


def source():return dict(runtime=runtime_source_manifest(),scripts={n:sha256(ROOT/'scripts'/n) for n in SCRIPTS})


def positive(value):
    if type(value) not in (float,int) or not math.isfinite(value) or value<=0:raise ValueError('Positive finite budget required')
    return value


def prepare(config,corpus_root,suites,*,driver_python,worker_python,release=None,setup_budget_s=600,kernel_budget_s=600,diagnostic_device=None):
    c=validate_probe_config(read(config));root=Path(corpus_root).resolve();files={}
    for value in (driver_python,worker_python):
        if not Path(value).is_absolute() or not Path(value).is_file():raise ValueError('Explicit existing Python executable required')
    if not 10<=kernel_budget_s<=3600:raise ValueError('Kernel budget must be 10..3600 seconds')
    positive(setup_budget_s)
    if c['mode']=='gpu' and release is None:raise ValueError('GPU workflow requires the installed immutable release')
    if c['mode']=='gpu' and (not isinstance(diagnostic_device,str) or not diagnostic_device.isdigit()):
        raise ValueError('GPU workflow requires one explicit diagnostic CUDA device index')
    if set(suites)-{'smoke','pilot','profile'} or 'smoke' not in suites:raise ValueError('Matching smoke suite required')
    if 'profile' in suites and 'pilot' not in suites:raise ValueError('Profile workflow requires a pilot stage')
    def track(p):
        p=Path(p).resolve();files[str(p)]=sha256(p);return str(p)
    config=track(config);definitions={};steps=[dict(id='environment',kind='environment',timeout_s=setup_budget_s),
                                            dict(id='model-preflight',kind='model',timeout_s=setup_budget_s)]
    corpus_id=None
    for stage in ('smoke','pilot','profile'):
        if stage not in suites:continue
        path=track(suites[stage]);s=validate_suite(read(path))
        if s['config']!=c or not s['executable'] or any(x['stage']!=stage for x in s['cases']):raise ValueError('Suite/config/stage not ready')
        book,_=load_corpus(s['corpus'],root,s['split'],c['mode'])
        if corpus_id and book['corpus_id']!=corpus_id:raise ValueError('Workflow mixes corpora')
        corpus_id=book['corpus_id']
        for row in book['rows']:
            for name in ('observation.json','arrays.npz'):track(root/row['directory']/name)
        definitions[stage]=path
        for i,case in enumerate(s['cases']):
            protocol=case['protocol'];timeout=protocol['warmup_max_s']+protocol['measure_max_s']+2*(c['drain_timeout_s']+2*c['request_timeout_s'])+5+setup_budget_s
            steps.append(dict(id=f'{stage}-{i}',kind='probe',stage=stage,case_id=case['case_id'],timeout_s=timeout))
        if stage=='smoke':steps.append(dict(id='kernel',kind='kernel',timeout_s=kernel_budget_s+15))
    plan=dict(schema_version='1.0',mode=c['mode'],config=config,corpus_root=str(root),suites=definitions,
        driver_python=str(Path(driver_python).absolute()),worker_python=str(Path(worker_python).absolute()),
        python_binary_hashes={str(Path(p).absolute()):sha256(Path(p).resolve()) for p in (driver_python,worker_python)},
        release=str(Path(release).resolve()) if release else None,diagnostic_device=diagnostic_device,
        setup_budget_s=setup_budget_s,kernel_budget_s=kernel_budget_s,source=source(),input_files=files,steps=steps,
        cleanup_allowance_per_step_s=10,scope='local installed Pod or CPU workspace; no SSH, package installation or Pod lifecycle actions')
    plan['plan_id']=digest(plan);return plan


def check(plan):
    expected=prepare(plan['config'],plan['corpus_root'],plan['suites'],driver_python=plan['driver_python'],worker_python=plan['worker_python'],
        release=plan['release'],setup_budget_s=plan['setup_budget_s'],kernel_budget_s=plan['kernel_budget_s'],diagnostic_device=plan['diagnostic_device'])
    if expected!=plan:raise ValueError('Workflow source, inputs or executable changed; prepare a new plan/run')
    if plan['release']:
        from .pod_transfer import MANIFEST,verify_installed
        release=Path(plan['release']);m=read(release/MANIFEST);verify_installed(release,m)
        if release!=ROOT.resolve() or m['kind']!='release':raise ValueError('Execute from the verified installed release')
    return dict(status='ready',plan_id=plan['plan_id'],steps=plan['steps'],maximum_total_wall_s=sum(x['timeout_s']+10 for x in plan['steps']),
        network_used=False,models_loaded=False,release_verified=bool(plan['release']))


def command(plan,step,output,completed,address):
    scripts=ROOT/'scripts';out=Path(output);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');c=plan['config']
    if step['kind']=='environment':args=[plan['worker_python'],str(scripts/'runtime_evidence.py'),'collect','--torch','--output',str(out/'environment.json')]
    elif step['kind']=='model':args=[plan['worker_python'],str(scripts/'postpod_models.py'),'preflight','--config',c,'--output',str(out/'preflight.json')]
    elif step['kind']=='kernel':
        args=[plan['worker_python'],str(scripts/'runtime_evidence.py'),'probe','--output',str(out/'kernel'),'--timeout',str(plan['kernel_budget_s'])]
        if plan['mode']=='cpu_dummy':args+=['--cpu-self-test']
        else:
            args+=['--config',c,'--corpus',str(out/'corpus.json'),'--root',plan['corpus_root']]
            write_json(out/'corpus.json',read(plan['suites']['smoke'])['corpus']);env['CUDA_VISIBLE_DEVICES']=plan['diagnostic_device']
    else:
        args=[plan['driver_python'],str(scripts/'postpod_models.py'),'run','--suite',plan['suites'][step['stage']],
            '--corpus-root',plan['corpus_root'],'--output',str(out/'probe'),'--case-id',step['case_id'],'--max-new','1']
        if plan['mode']=='gpu':
            args+=['--address',address]
            if step['stage']!='smoke':args+=['--smoke-evidence',completed['smoke-0']['raw_directory']]
    return args,env


def supervise(argv,env,out,timeout):
    started=time.monotonic();timed_out=interrupted=False
    with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
        proc=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            timed_out=isinstance(exc,subprocess.TimeoutExpired);interrupted=not timed_out
            for sig in (signal.SIGTERM,signal.SIGKILL):
                try:os.killpg(proc.pid,sig)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            code=proc.wait(timeout=1)
    return dict(exit_code=code,timed_out=timed_out,interrupted=interrupted,elapsed_s=time.monotonic()-started)


def validate_output(plan,step,out):
    if step['kind']=='environment':
        r=read(out/'environment.json')
        if r['source_manifest']!=runtime_source_manifest():raise ValueError('Worker environment source differs')
    elif step['kind']=='model':
        r=read(out/'preflight.json')
        if r['status']!='passed' or r['config_sha256']!=digest(read(plan['config'])):raise ValueError('Model preflight failed')
    elif step['kind']=='kernel':
        verify_evidence(out/'kernel');r=read(out/'kernel/result.json')
        if r['status']!='complete' or r['process_exit_code']!=0:raise ValueError('Kernel diagnostic incomplete')
        if plan['mode']=='gpu' and not r.get('gpu_kernel_events_present'):raise ValueError('GPU kernel trace missing')
    else:
        paths=list((out/'probe').glob('*/attempt-*/raw'))
        if len(paths)!=1:raise ValueError('Expected exactly one attempted probe case')
        p=paths[0];m=reaudit(p);r=read(p/'result.json');x=read(p/'condition.json')
        if x['case']['case_id']!=step['case_id'] or r['status']!='complete' or not r['drained'] or m['successful']!=m['arrivals'] or not m['successful']:
            raise ValueError('Probe shortfall or request failure')
        return dict(raw_directory=str(p.resolve()))
    return {}


def run(plan,output,*,execute=False,max_new=1,budget_s=None,address=None,cleanup_review=None):
    checked=check(plan)
    if not execute:return dict(checked,status='planned',executed_steps=0)
    positive(budget_s)
    if type(max_new) is not int or max_new<1:raise ValueError('max-new must be positive')
    if plan['mode']=='gpu' and (not address or address=='local'):raise ValueError('Explicit existing Ray address required')
    if plan['mode']=='cpu_dummy' and address:raise ValueError('CPU workflow refuses external Ray address')
    output=Path(output).resolve();done={};rows=[];started=time.monotonic();count=0
    matrix=dict(plan=plan,ray_address=address)
    with ExperimentStore(output,matrix) as store:
        for step in plan['steps']:
            sid=step['id']
            latest=store.db.execute('SELECT number,status FROM attempts WHERE case_id=? ORDER BY number DESC LIMIT 1',(sid,)).fetchone()
            if plan['mode']=='gpu' and latest and latest[1] in ('interrupted','failed'):
                abandoned=output/sid/f'attempt-{latest[0]:03d}'
                required=digest(dict(plan_id=plan['plan_id'],step=sid,attempt=latest[0],
                    files={str(f.relative_to(abandoned)):sha256(f) for f in sorted(abandoned.rglob('*')) if f.is_file()}))
                if not cleanup_review or cleanup_review.get('attempt_sha256')!=required or cleanup_review.get('cleanup_confirmed') is not True or not all(
                    isinstance(cleanup_review.get(k),str) and cleanup_review[k].strip() for k in ('reviewer','reason')):
                    return dict(status='blocked_cluster_cleanup',steps=rows,attempt_sha256=required,executed_steps=count)
            if latest and latest[1]=='complete' and store.completed(sid):
                number=store.db.execute("SELECT MAX(number) FROM attempts WHERE case_id=? AND status='complete'",(sid,)).fetchone()[0]
                previous=output/sid/f'attempt-{number:03d}';old=read(previous/'result.json')
                if old['status']=='passed':
                    validate_output(plan,step,previous);done[sid]=old;rows.append(dict(step=sid,status='skipped'));continue
                if old.get('needs_cluster_cleanup'):
                    required=sha256(previous/'artifact_hashes.json')
                    if not cleanup_review or cleanup_review.get('attempt_sha256')!=required or cleanup_review.get('cleanup_confirmed') is not True or not all(
                        isinstance(cleanup_review.get(k),str) and cleanup_review[k].strip() for k in ('reviewer','reason')):
                        return dict(status='blocked_cluster_cleanup',steps=rows,attempt_sha256=required,executed_steps=count)
            if count>=max_new:return dict(status='step_limit',steps=rows,executed_steps=count)
            if time.monotonic()-started+step['timeout_s']+10>budget_s:return dict(status='budget_stop',next_step=sid,steps=rows,executed_steps=count)
            with store.attempt(sid) as out:
                result=dict(status='failed',step=sid,needs_cluster_cleanup=False)
                try:
                    argv,env=command(plan,step,out,done,address)
                    write_json(out/'command.json',dict(argv=argv,timeout_s=step['timeout_s']))
                    if cleanup_review:write_json(out/'cleanup_review.json',cleanup_review)
                    process=supervise(argv,env,out,step['timeout_s']);result['process']=process
                    result['needs_cluster_cleanup']=plan['mode']=='gpu' and (process['timed_out'] or process['interrupted'] or process['exit_code']<0)
                    if process['exit_code']!=0 or process['timed_out'] or process['interrupted']:raise ValueError('Stage process failed/interrupted/timed out')
                    result.update(validate_output(plan,step,out),status='passed')
                except Exception as exc:result.update(error_type=type(exc).__name__,reason=str(exc))
                write_json(out/'result.json',result)
            count+=1;rows.append(dict(step=sid,status=result['status'],directory=str(out)))
            if result['status']!='passed':return dict(status='failed',steps=rows,executed_steps=count,needs_cluster_cleanup=result['needs_cluster_cleanup'])
            done[sid]=result
    return dict(status='complete',steps=rows,executed_steps=count,real_GPU_verified=plan['mode']=='gpu',profile_adoption_still_requires_review=True)
