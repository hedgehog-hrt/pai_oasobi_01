"""SLO-free, one-worker R05 replay contracts. Operational watchdogs are not SLOs."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math
from .contracts import obj,TEXT,POSITIVE,validate
from .execution_plan import digest,runtime_source_manifest,validate_binding,DUMMY_FACTORY
from .profile_protocol import template as model_template,load_corpus
from .paper_conditions import validate as validate_paper
from .regression import MODEL_REVISION

CONFIG_SCHEMA=obj(dict(schema_version={'const':'1.0'},mode={'enum':['gpu','cpu_dummy']},model_revision=TEXT,
    runtime_fingerprint=TEXT,hardware_kind=TEXT,binding={'type':'object'},batch_wait_ms={'type':'number','minimum':0,'maximum':1000},
    watchdog_s={'type':'number','exclusiveMinimum':0,'maximum':600},drain_s={'type':'number','exclusiveMinimum':0,'maximum':600},
    setup_budget_s={'type':'number','exclusiveMinimum':0,'maximum':3600}))


def template():
    old=model_template('s1','groot')
    return dict(schema_version='1.0',**{k:old[k] for k in ('mode','model_revision','runtime_fingerprint','hardware_kind','binding')},
                batch_wait_ms=5.,watchdog_s=120.,drain_s=150.,setup_budget_s=600.)


def config(c,ready=True):
    validate(c,CONFIG_SCHEMA);validate_binding(c['binding'],c['mode'])
    if c['drain_s']<c['watchdog_s']:raise ValueError('Drain budget must cover operational watchdog')
    if ready and ('SET_' in json.dumps(c) or '/SET/' in json.dumps(c)):raise ValueError('Resolve runtime/model configuration')
    if c['mode']=='gpu':
        b=c['binding'];kw=b['kwargs']
        if b['factory']!='rosa_repro.model_backends:groot_backend' or c['model_revision']!=MODEL_REVISION:raise ValueError('Pinned S1 GR00T required')
        if ready and kw.get('stateless_verified') is not True:raise ValueError('Verify stateless S1 before sharing')
        if any(not Path(kw[k]).is_absolute() for k in ('repo','model_path')):raise ValueError('Absolute model/repository paths required')
    return c


def timing(phase,*,warmup_s=None,duration_s=None,warmup_min_success=None,measure_min_success=None,max_requests=100000):
    defaults={'cpu_smoke':(.5,1.,1,1),'pilot':(5.,10.,5,10),'formal':(30.,120.,20,1000)}
    if phase not in defaults:raise ValueError('Unknown R05 timing phase')
    dw,dd,dwc,dmc=defaults[phase]
    w=dw if warmup_s is None else warmup_s;d=dd if duration_s is None else duration_s
    wc=dwc if warmup_min_success is None else warmup_min_success;mc=dmc if measure_min_success is None else measure_min_success
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (w,d)) or w>3600 or d>86400:raise ValueError('Invalid fixed window')
    if any(type(v) is not int or not 1<=v<=100000 for v in (wc,mc,max_requests)):raise ValueError('Invalid success/request count')
    if phase!='cpu_smoke' and (w<dw or d<dd or wc<dwc or mc<dmc):raise ValueError('Scientific minimum cannot be lowered')
    if max_requests<wc+mc:raise ValueError('Request budget cannot cover required samples')
    return dict(phase=phase,warmup_s=w,duration_s=d,warmup_min_success=wc,measure_min_success=mc,max_requests=max_requests,
                fixed_windows=True,early_stop_on_warmup_shortfall=True,latency_slo_ms=None)


def plan_for(c,cell):
    config(c)
    if cell['experiment']!='R05' or cell['latency_slo_ms'] is not None or cell['slo_policy']!='none' or cell['frequency_cap_hz'] is not None:raise ValueError('SLO-free R05 cell required')
    if type(cell['batch_size']) is not int or cell['batch_size'] not in (1,2,4,8,16) or cell['robots'] not in (1,4,16):raise ValueError('Unresolved batch/robot count')
    w=dict(worker_id='r05-s1',component_id='s1',role='s1',model_revision=c['model_revision'],runtime_fingerprint=c['runtime_fingerprint'],
        hardware_kind=c['hardware_kind'],expected_memory_mib=1,profile_id='r05-unprofiled',profile_source='unmeasured',
        batch_size=cell['batch_size'],batch_wait_ms=c['batch_wait_ms'],resources=dict(num_cpus=1,num_gpus=int(c['mode']=='gpu')),
        members=[f'r{i}' for i in range(cell['robots'])],backend=deepcopy(c['binding']))
    result=dict(purpose='r05_slo_free_saved_observation_replay',mode=c['mode'],config=deepcopy(c),cell=deepcopy(cell),workers=[w],
                latency_slo_ms=None,frequency_cap_hz=None,action_semantics='virtual_timed_chunks')
    result['plan_id']=digest(result);return result


def validate_plan(p):
    if p!=plan_for(p['config'],p['cell']):raise ValueError('R05 plan modified')
    return p


def prepare(paper,c,corpus,*,stage,groups=None,protocol=None,prediction_artifacts=None):
    validate_paper(paper);config(c)
    if c['mode']=='gpu' and corpus.get('kind')!='saved_robot_observations':raise ValueError('GPU requires real saved-observation corpus')
    if stage not in ('pilot','confirmation'):raise ValueError('R05 stage required')
    protocol=protocol or timing('cpu_smoke' if c['mode']=='cpu_dummy' else 'pilot' if stage=='pilot' else 'formal')
    expected=timing(protocol['phase'],**{k:protocol[k] for k in ('warmup_s','duration_s','warmup_min_success','measure_min_success','max_requests')})
    if protocol!=expected or (protocol['phase']=='cpu_smoke')!=(c['mode']=='cpu_dummy') or (c['mode']=='gpu' and stage=='confirmation' and protocol['phase']!='formal'):
        raise ValueError('Mode/stage/timing mismatch')
    known={x['group'] for x in paper['cells'] if x['experiment']=='R05'}
    groups=sorted(known) if groups is None else sorted(groups)
    if not groups or len(groups)!=len(set(groups)) or set(groups)-known:raise ValueError('Invalid R05 group selection')
    candidates={x['id']:deepcopy(x) for x in paper['cells'] if x['experiment']=='R05' and x['stage']==stage and x['group'] in groups}
    cells=[candidates[cid] for cid in paper['randomized_order'] if cid in candidates]
    if any(x['batch_size'] is None for x in cells):raise ValueError('Resolve every predicted confirmation candidate first')
    artifacts=deepcopy(prediction_artifacts or {})
    if stage=='confirmation':
        for g in groups:
            selection=paper['inputs']['r05_selection'].get(g)
            if not selection:raise ValueError('Predicted candidate and evidence required for each confirmation group')
            h=selection['prediction_evidence_sha256'];text=artifacts.get(h)
            if not isinstance(text,str) or hashlib.sha256(text.encode()).hexdigest()!=h:raise ValueError('Prediction evidence missing or modified')
            pred=json.loads(text)
            if pred.get('kind')!='r05_batch_prediction' or pred.get('config_sha256')!=digest(c) or pred.get('corpus_id')!=corpus['corpus_id'] or pred.get('groups',{}).get(g)!=selection['batch_size']:
                raise ValueError('Prediction model/input/batch binding differs')
            if not isinstance(pred.get('method'),str) or not pred['method'].strip() or not pred.get('source_evidence_sha256'):raise ValueError('Prediction method and source evidence required')
            if pred.get('method') not in ('measured_closed_loop_mean_cycle_v1','measured_load_curve_fixed_point_v1') or pred.get('source_manifest')!=runtime_source_manifest() or not isinstance(pred.get('inputs',{}).get('pilot_directories'),dict):raise ValueError('Generated current-source prediction required')
            for value in pred['source_evidence_sha256']:
                if not isinstance(value,str) or len(value)!=64 or any(ch not in '0123456789abcdef' for ch in value):raise ValueError('Invalid prediction source hash')
    elif artifacts:raise ValueError('Pilot does not consume prediction selections')
    blockers=[]
    for cell in cells:
        plan_for(c,cell)
        # Even zero inference time cannot complete more virtual actions than this bound.
        upper=cell['robots']*max(0,math.ceil(protocol['duration_s']/cell['action_time_s'])-1)
        if upper<protocol['measure_min_success']:blockers.append(dict(cell_id=cell['id'],reason='action_only_upper_bound_below_sample_minimum',upper_bound=upper))
    suite=dict(schema_version='1.0',kind='r05_slo_free_suite',paper=deepcopy(paper),config=deepcopy(c),corpus=deepcopy(corpus),split='profile',
        stage=stage,groups=groups,protocol=protocol,cells=cells,prediction_artifacts=artifacts,source_manifest=runtime_source_manifest(),
        executable=not blockers,blocking_reasons=blockers,action_semantics='virtual_timed_chunks',latency_slo_ms=None,
        phase_budget_s=protocol['warmup_s']+protocol['duration_s']+2*(c['drain_s']+1),
        measurement_selection='completed virtual action chunks within fixed window; no latency acceptance threshold')
    suite['suite_id']=digest(suite);return suite


def validate_suite(s,root=None):
    expected=prepare(s['paper'],s['config'],s['corpus'],stage=s['stage'],groups=s['groups'],protocol=s['protocol'],prediction_artifacts=s['prediction_artifacts'])
    if s!=expected:raise ValueError('R05 suite/source/conditions changed')
    if root is not None:load_corpus(s['corpus'],root,s['split'],s['config']['mode'])
    return s
