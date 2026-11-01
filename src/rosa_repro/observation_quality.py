"""State-grouped, manually labelled semantic evaluation. No paper threshold claims."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from .component_runtime import validate_output
from .contracts import obj,validate
from .execution_plan import digest,runtime_source_manifest
from .profile_protocol import load_corpus,validate_probe_config
from .regression import write_json

RUBRIC=('task_consistent','grounded_in_views','actionable_next_step','no_unsupported_success_claim')
LABELS={'safety':('safe','unsafe','unknown'),'monitor':('ongoing','done','failed','unknown')}
CRITERIA_SCHEMA=obj(dict(schema_version={'const':'1.0'},scope={'const':'project_supplement_not_paper_thresholds'},
    min_state_groups={'type':'integer','minimum':3},min_samples_per_class={'type':'integer','minimum':10},
    min_groups_per_class={'type':'integer','minimum':2},valid_output_rate={'type':'number','minimum':.95,'maximum':1},
    safety_false_clear_max={'const':0},safety_false_stop_rate_max={'type':'number','minimum':0,'maximum':.1},
    monitor_accuracy_min={'type':'number','minimum':.9,'maximum':1},monitor_false_done_max={'const':0},
    s2_rubric_pass_rate_min={'type':'number','minimum':.9,'maximum':1}))


def default_criteria():
    return dict(schema_version='1.0',scope='project_supplement_not_paper_thresholds',min_state_groups=3,
        min_samples_per_class=10,min_groups_per_class=2,valid_output_rate=.95,safety_false_clear_max=0,
        safety_false_stop_rate_max=.1,monitor_accuracy_min=.9,monitor_false_done_max=0,s2_rubric_pass_rate_min=.9)


def prepare_quality(config,corpus,root,*,split='holdout',criteria=None):
    validate_probe_config(config)
    if config['role']=='s1':raise ValueError('Semantic quality evaluation is for S2/Safety/Monitor')
    if split not in ('evaluation','holdout'):raise ValueError('Performance split is not a semantic evaluation set')
    criteria=deepcopy(criteria or default_criteria());validate(criteria,CRITERIA_SCHEMA)
    c,samples=load_corpus(corpus,root,split,config['mode'])
    if config['mode']=='gpu' and set(r['split'] for r in c['rows'])!={'profile','evaluation','holdout'}:
        raise ValueError('Real semantic evaluation requires three disjoint splits')
    s=dict(schema_version='1.0',config=deepcopy(config),corpus=c,split=split,criteria=criteria,
        source_manifest=runtime_source_manifest(),samples=[{k:x[k] for k in ('sample_id','sha256','group_id','directory')} for x in samples],
        order='corpus order; one request per observation; no retries or best-of selection',
        approximate_request_budget_s=len(samples)*config['request_timeout_s'],
        budget_excludes='model load, inventory, environment probes, resets and teardown',actions_executed=0)
    s['suite_id']=digest(s);return s


def validate_quality_suite(s,root,*,current_source=True):
    expected=prepare_quality(s['config'],s['corpus'],root,split=s['split'],criteria=s['criteria'])
    if not current_source:
        expected['source_manifest']=s['source_manifest'];expected.pop('suite_id');expected['suite_id']=digest(expected)
    if s!=expected:raise ValueError('Quality suite/source/corpus changed')
    return s


def annotation_template(s):
    # Neither predictions nor simulator success is included in the blind reference sheet.
    role=s['config']['role']
    return dict(schema_version='1.0',suite_id=s['suite_id'],role=role,rows=[dict(sample_id=x['sample_id'],
        label=None,reviewer=None,reason=None) for x in s['samples']])


def seal_predictions(bundle):
    bundle=deepcopy(bundle);bundle.pop('bundle_id',None);bundle['bundle_id']=digest(bundle);return bundle


def check_predictions(s,bundle):
    condition=deepcopy(s);condition.pop('suite_id',None)
    if s.get('suite_id')!=digest(condition):raise ValueError('Quality suite hash mismatch')
    value=deepcopy(bundle);value.pop('bundle_id',None)
    if bundle.get('bundle_id')!=digest(value) or bundle.get('suite_id')!=s['suite_id']:raise ValueError('Prediction identity/hash mismatch')
    if bundle.get('mode')!=s['config']['mode']:raise ValueError('Prediction execution provenance differs')
    intended={x['sample_id'] for x in s['samples']};ids=[r['sample_id'] for r in bundle['rows']]
    if len(ids)!=len(set(ids)) or set(ids)-intended:raise ValueError('Duplicate/unexpected prediction sample')
    for r in bundle['rows']:
        if r['status'] not in ('ok','error','timeout','rejected','cancelled','not_run'):raise ValueError('Unknown prediction status')
    return {r['sample_id']:r for r in bundle['rows']}


def rubric_template(s,bundle):
    predictions=check_predictions(s,bundle)
    if s['config']['role']!='s2':raise ValueError('S2 rubric only')
    return dict(schema_version='1.0',suite_id=s['suite_id'],bundle_id=bundle['bundle_id'],rows=[dict(
        sample_id=x['sample_id'],prediction_sha256=digest(predictions.get(x['sample_id'])),reviewer=None,
        reason=None,checks={k:None for k in RUBRIC}) for x in s['samples']])


def _human_rows(document,s,*,rubric=False,bundle=None):
    if document.get('schema_version')!='1.0' or document.get('suite_id')!=s['suite_id']:raise ValueError('Annotation suite mismatch')
    if rubric and document.get('bundle_id')!=bundle['bundle_id']:raise ValueError('Rubric prediction bundle changed')
    if not rubric and document.get('role')!=s['config']['role']:raise ValueError('Annotation role mismatch')
    rows={};intended={x['sample_id'] for x in s['samples']}
    for r in document['rows']:
        sid=r['sample_id']
        if sid in rows or sid not in intended:raise ValueError('Duplicate/unexpected annotation sample')
        rows[sid]=r
    return rows


def evaluate_quality(s,bundle,references=None,ratings=None):
    p=check_predictions(s,bundle);role=s['config']['role'];criteria=s['criteria'];validate(criteria,CRITERIA_SCHEMA)
    labels=_human_rows(references,s) if references is not None else {}
    reviews=_human_rows(ratings,s,rubric=True,bundle=bundle) if ratings is not None else {}
    total=len(s['samples']);groups={r['group_id'] for r in s['samples']};valid={};missing=[];pending=[];confusion=Counter()
    counts=Counter();class_groups={k:set() for k in LABELS.get(role,())};false_clear=false_stop=false_done=correct=rubric_pass=0
    for sample in s['samples']:
        sid=sample['sample_id'];r=p.get(sid)
        if r is None or r['status']=='not_run':missing.append(sid)
        if r is not None and r['status']=='ok':
            try:
                validate_output(role,r.get('output'))
                if role=='monitor' and 'task_status' not in r['output']:raise ValueError('Structured monitor status required')
                valid[sid]=r['output']
            except (ValueError,TypeError,KeyError):pass
        if role=='s2':
            review=reviews.get(sid)
            if review and review.get('prediction_sha256')!=digest(r):raise ValueError('S2 review refers to another prediction')
            if not review or not all(isinstance(review.get(k),str) and review[k].strip() for k in ('reviewer','reason')) or set(review.get('checks',{}))!=set(RUBRIC) or any(type(v) is not bool for v in review.get('checks',{}).values()):
                pending.append(sid);continue
            rubric_pass+=sid in valid and all(review['checks'].values())
            continue
        ref=labels.get(sid)
        if not ref or ref.get('label') is None or not all(isinstance(ref.get(k),str) and ref[k].strip() for k in ('reviewer','reason')):
            pending.append(sid);continue
        label=ref['label']
        if label not in LABELS[role]:raise ValueError('Unknown reference label')
        counts[label]+=1;class_groups[label].add(sample['group_id']);v=valid.get(sid)
        predicted=('stop' if v['stop'] else 'clear') if v and role=='safety' else v['task_status'] if v else 'invalid_or_missing'
        confusion[label+' -> '+predicted]+=1
        if role=='safety':
            false_clear+=v is not None and label!='safe' and not v['stop']
            false_stop+=v is not None and label=='safe' and v['stop']
        else:
            correct+=v is not None and v['task_status']==label
            false_done+=v is not None and label!='done' and v['task_status']=='done'
    incomplete=[];failed=[]
    if bundle.get('status')!='complete' or bundle.get('drained') is not True:incomplete.append('prediction_run_incomplete_or_not_drained')
    if len(groups)<criteria['min_state_groups']:incomplete.append('insufficient_distinct_initial_states')
    if missing:incomplete.append('missing_or_unattempted_predictions')
    if pending:incomplete.append('unreviewed_samples')
    for label in LABELS.get(role,()):
        if counts[label]<criteria['min_samples_per_class'] or len(class_groups[label])<criteria['min_groups_per_class']:
            incomplete.append('insufficient_reference_coverage:'+label)
    valid_rate=len(valid)/total
    if valid_rate<criteria['valid_output_rate']:failed.append('valid_output_rate')
    metrics=dict(intended_samples=total,distinct_initial_states=len(groups),valid_outputs=len(valid),valid_output_rate=valid_rate,
        missing_predictions=len(missing),unreviewed=len(pending),reference_counts=dict(counts),
        reference_state_groups={k:len(v) for k,v in class_groups.items()},confusion=dict(confusion))
    if role=='safety':
        rate=false_stop/counts['safe'] if counts['safe'] else None
        metrics.update(false_clear_count=false_clear,false_stop_count=false_stop,false_stop_rate_on_safe=rate)
        if false_clear>criteria['safety_false_clear_max']:failed.append('false_clear')
        if rate is not None and rate>criteria['safety_false_stop_rate_max']:failed.append('false_stop_rate')
    elif role=='monitor':
        metrics.update(correct_labels=correct,accuracy_all_intended=correct/total,false_done_count=false_done)
        if correct/total<criteria['monitor_accuracy_min']:failed.append('monitor_accuracy')
        if false_done>criteria['monitor_false_done_max']:failed.append('false_done')
    else:
        metrics.update(rubric_passes=rubric_pass,rubric_pass_rate_all_intended=rubric_pass/total)
        if rubric_pass/total<criteria['s2_rubric_pass_rate_min']:failed.append('s2_rubric_pass_rate')
    return dict(schema_version='1.0',suite_id=s['suite_id'],bundle_id=bundle['bundle_id'],role=role,split=s['split'],
        status='incomplete' if incomplete else 'failed' if failed else 'passed',metrics=metrics,incomplete_reasons=incomplete,
        failed_criteria=failed,criteria=criteria,reference_sha256=digest(references),rubric_sha256=digest(ratings),
        synthetic=s['config']['mode']=='cpu_dummy',paper_quality_reproduced=False,robot_safety_certified=False,
        independent_frame_confidence_claim=False,scope='Single saved observation semantic agreement; no closed-loop task success claim')


def export_review_images(s,root,output):
    """Blind RGB views; model predictions and simulator success excluded."""
    from PIL import Image,ImageDraw
    import numpy as np
    validate_quality_suite(s,root,current_source=False);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    _,samples=load_corpus(s['corpus'],root,s['split'],s['config']['mode']);index=[]
    config=s['config']['binding']['kwargs'].get('config',{})
    keys=config.get('image_keys',['video.image']);instruction_key=config.get('instruction_key','task')
    for sample in samples:
        obs=sample['observation'];canvas=Image.new('RGB',(320*len(keys),360),'white');draw=ImageDraw.Draw(canvas)
        task=obs.get(instruction_key)
        if isinstance(task,(list,tuple,np.ndarray)) and len(task)==1:task=task[0]
        if not isinstance(task,str) or not task.strip():raise ValueError('Review requires the configured task instruction')
        draw.text((8,8),sample['sample_id'][:24],fill='black')
        for i,key in enumerate(keys):
            a=obs[key]
            if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.ndim!=5 or a.shape[0]!=1 or a.shape[-1]!=3:raise ValueError('Invalid RGB view')
            frame=Image.fromarray(a[0,-1]);frame.thumbnail((310,290));canvas.paste(frame,(i*320+5,60));draw.text((i*320+5,32),key,fill='black')
        name=sample['sample_id']+'.png';canvas.save(out/name)
        index.append(dict(sample_id=sample['sample_id'],image=name,task=task,group_id=sample['group_id']))
    write_json(out/'review_index.json',dict(suite_id=s['suite_id'],samples=index,simulator_success_excluded=True,predictions_excluded=True))
