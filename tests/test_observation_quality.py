import asyncio,json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from test_paired import bank
from test_profile_preparation import config,Transport
from rosa_repro.capture_plan import default_design,make_capture_plan,validate_capture_plan,capture
from rosa_repro.dummy import DummyEpisode,DummyTransport,observation
from rosa_repro.execution_plan import digest
from rosa_repro.paired import save_state_bank,digest_value
from rosa_repro.profile_protocol import build_corpus,load_corpus,probe_plan
from rosa_repro.observation_quality import (prepare_quality,annotation_template,rubric_template,evaluate_quality,
    seal_predictions,export_review_images,RUBRIC)
from rosa_repro.quality_runner import run_quality
from rosa_repro.regression import save_observation


def collected(tmp_path,**changes):
    b=bank(tmp_path/'bank',6);d=dict(default_design(),states_per_task=6,capture_every_chunks=1,**changes)
    p=make_capture_plan([b],d);out=tmp_path/'capture'
    r=capture(p,[b],out,lambda task:DummyEpisode(),lambda:DummyTransport('official'),execution_kind='cpu_dummy')
    return b,p,out,r


def test_capture_is_state_disjoint_bounded_and_preserves_terminal_frames(tmp_path):
    b,p,out,r=collected(tmp_path);validate_capture_plan(p,[b]);c=build_corpus(out,kind='synthetic_cpu')
    assert len(p['cases'])==r['completed_episodes']==6 and r['observations']==18
    assert set(x['split'] for x in c['rows'])=={'profile','evaluation','holdout'}
    for split in ('profile','evaluation','holdout'):load_corpus(c,out,split,'cpu_dummy')
    frames=[json.loads(x.read_text())['provenance'] for x in out.rglob('observation.json')]
    assert {x['capture_stage'] for x in frames}=={'initial','progress','terminal'}
    assert all(x['simulator_success'] is True for x in frames if x['capture_stage']=='terminal')
    assert not r['quality_labels_assigned'] and not r['safety_hazard_coverage_guaranteed']
    assert len(list(out.rglob('action-*.npz')))==12<=p['limits']['action_calls_upper']
    c['rows'][0]['split']='holdout' if c['rows'][0]['split']!='holdout' else 'evaluation'
    c.pop('corpus_id');c['corpus_id']=digest(c)
    with pytest.raises(ValueError):load_corpus(c,out,'profile','cpu_dummy')


def test_capture_includes_policy_errors_and_does_not_label_step_limit_as_failure(tmp_path):
    b=bank(tmp_path/'bank',6);p=make_capture_plan([b],dict(default_design(),states_per_task=6,max_steps=1))
    def failed():
        t=DummyTransport('official')
        def fail(*a):raise TimeoutError('fixture')
        t.predict=fail;return t
    r=capture(p,[b],tmp_path/'fail',lambda t:DummyEpisode(),failed,execution_kind='cpu_dummy')
    assert r['started_episodes']==6 and r['completed_episodes']==0 and r['observations']==6
    assert all(o['status']=='failed' for o in r['outcomes'])
    r=capture(p,[b],tmp_path/'limit',lambda t:DummyEpisode(),lambda:DummyTransport('official'),execution_kind='cpu_dummy')
    assert all(o['termination']=='step_limit' and not o['simulator_success'] for o in r['outcomes'])
    assert not r['quality_labels_assigned']


def test_action_chunk_budget_also_holds_for_short_environment_steps(tmp_path):
    b=bank(tmp_path/'bank',6);p=make_capture_plan([b],dict(default_design(),states_per_task=6,max_steps=9))
    class Short(DummyEpisode):
        def step(self,a,n):return dict(observation=observation(self.value),steps=1,success=False,done=False)
    r=capture(p,[b],tmp_path/'short',lambda t:Short(),lambda:DummyTransport('official'),execution_kind='cpu_dummy')
    assert sum(o['action_chunks'] for o in r['outcomes'])==p['limits']['action_calls_upper']==12
    assert all(o['termination']=='chunk_budget' for o in r['outcomes'])


def test_no_actions_when_initial_state_already_successful(tmp_path):
    b=bank(tmp_path/'bank',6);p=make_capture_plan([b],dict(default_design(),states_per_task=6))
    class Success(DummyEpisode):
        def reset(self,s,seed):
            result=super().reset(s,seed);self.last_success=True;return result
    r=capture(p,[b],tmp_path/'success',lambda t:Success(),lambda:DummyTransport('official'),execution_kind='cpu_dummy')
    assert all(o['action_chunks']==0 and o['termination']=='success_at_reset' for o in r['outcomes'])


@pytest.mark.parametrize('bad',['insufficient','duplicate','mutated','synthetic_as_real'])
def test_invalid_capture_inputs_refused_before_environment_creation(tmp_path,bad):
    b=bank(tmp_path/'bank',6);d=dict(default_design(),states_per_task=6);p=make_capture_plan([b],d)
    if bad=='insufficient':d['states_per_task']=30
    if bad=='duplicate':
        b=tmp_path/'duplicate';save_state_bank(b,[np.array([1.])]*6,task='CPU',source={'kind':'synthetic'})
    with pytest.raises(ValueError):
        if bad=='mutated':
            (b/'states.npz').write_bytes(b'changed');validate_capture_plan(p,[b])
        elif bad=='synthetic_as_real':capture(p,[b],tmp_path/'never',None,None,execution_kind='libero_gpu')
        else:make_capture_plan([b],d)
    assert not (tmp_path/'never').exists()


def quality_fixture(tmp_path,role='safety'):
    root=tmp_path/'images'
    # 4 classes, each in 3 independent groups, 4 frames per group.
    for g in range(12):
        for f in range(4):
            save_observation(root/f'{g:02d}'/str(f),observation(g),dict(task='CPU',case={'state_sha256':digest_value([g])},
                capture_split='holdout',execution_kind='cpu_dummy'))
    c=build_corpus(root,kind='synthetic_cpu');s=prepare_quality(config(role),c,root)
    labels=('safe','unsafe','unknown') if role=='safety' else ('ongoing','done','failed','unknown')
    refs=annotation_template(s);rows=[]
    for i,x in enumerate(s['samples']):
        label=labels[(i//4)%len(labels)]
        refs['rows'][i].update(label=label,reviewer='CPU synthetic fixture author',reason='Synthetic assertion only')
        output=dict(stop=label!='safe',reason='fixture') if role=='safety' else dict(task_status=label,text='fixture') if role=='monitor' else dict(plan='fixture')
        rows.append(dict(sample_id=x['sample_id'],status='ok',output=output))
    bundle=seal_predictions(dict(suite_id=s['suite_id'],mode='cpu_dummy',status='complete',drained=True,rows=rows))
    return root,s,refs,bundle

def test_review_export_uses_configured_libero_instruction(tmp_path):
    root=tmp_path/'images'
    for i in range(3):
        obs=observation(i);obs['annotation.human.action.task_description']=('Pick up the soup.',)
        save_observation(root/str(i),obs,dict(task='CPU',case={'state_sha256':digest_value([i])},
            capture_split='holdout',execution_kind='cpu_dummy'))
    c=config('safety');c['binding']['kwargs']['config']={'instruction_key':'annotation.human.action.task_description'}
    suite=prepare_quality(c,build_corpus(root,kind='synthetic_cpu'),root)
    export_review_images(suite,root,tmp_path/'review')
    index=json.loads((tmp_path/'review/review_index.json').read_text())
    assert all(row['task']=='Pick up the soup.' for row in index['samples'])


@pytest.mark.parametrize('role',['safety','monitor'])
def test_human_labelled_fixture_passes_but_unreviewed_labels_do_not(tmp_path,role):
    root,s,refs,b=quality_fixture(tmp_path,role)
    result=evaluate_quality(s,b,refs);assert result['status']=='passed' and result['synthetic']
    assert not result['paper_quality_reproduced'] and not result['robot_safety_certified']
    missing=evaluate_quality(s,b,annotation_template(s));assert missing['status']=='incomplete'
    assert missing['metrics']['unreviewed']==48


@pytest.mark.parametrize('label',['unsafe','unknown'])
def test_safety_false_clear_fails_even_with_high_overall_validity(tmp_path,label):
    _,s,refs,b=quality_fixture(tmp_path);i=next(i for i,x in enumerate(refs['rows']) if x['label']==label)
    b['rows'][i]['output']['stop']=False;b=seal_predictions(b)
    r=evaluate_quality(s,b,refs);assert r['status']=='failed' and r['metrics']['false_clear_count']==1


def test_monitor_false_done_invalid_and_missing_stay_in_denominator(tmp_path):
    _,s,refs,b=quality_fixture(tmp_path,'monitor');b['rows'][0]['output']['task_status']='done'
    b['rows'][1].update(status='error',output=None);b=seal_predictions(b)
    r=evaluate_quality(s,b,refs);assert r['status']=='failed' and r['metrics']['accuracy_all_intended']==46/48
    assert r['metrics']['false_done_count']==1
    b['rows'].pop();b=seal_predictions(b);r=evaluate_quality(s,b,refs)
    assert r['status']=='incomplete' and r['metrics']['intended_samples']==48


def test_class_coverage_cannot_be_replaced_by_repeated_easy_frames(tmp_path):
    _,s,refs,b=quality_fixture(tmp_path)
    for row in refs['rows']:row['label']='safe'
    r=evaluate_quality(s,b,refs)
    assert r['status']=='incomplete' and 'insufficient_reference_coverage:unsafe' in r['incomplete_reasons']


def test_s2_human_rubric_is_bound_to_exact_prediction(tmp_path):
    _,s,_,b=quality_fixture(tmp_path,'s2');ratings=rubric_template(s,b)
    assert evaluate_quality(s,b,ratings=ratings)['status']=='incomplete'
    for r in ratings['rows']:r.update(reviewer='synthetic author',reason='CPU fixture',checks={k:True for k in RUBRIC})
    assert evaluate_quality(s,b,ratings=ratings)['status']=='passed'
    b['rows'][0]['output']['plan']='changed';b=seal_predictions(b)
    with pytest.raises(ValueError):evaluate_quality(s,b,ratings=ratings)
    ratings['bundle_id']=b['bundle_id']
    with pytest.raises(ValueError):evaluate_quality(s,b,ratings=ratings)


@pytest.mark.parametrize('bad',['duplicate','changed','wrong_suite','unknown_label','undrained'])
def test_prediction_annotation_and_run_integrity(tmp_path,bad):
    _,s,refs,b=quality_fixture(tmp_path)
    if bad=='duplicate':b['rows'].append(b['rows'][0]);b=seal_predictions(b)
    if bad=='changed':b['rows'][0]['output']['stop']=True
    if bad=='wrong_suite':refs['suite_id']='another'
    if bad=='unknown_label':refs['rows'][0]['label']='invented'
    if bad=='undrained':
        b['drained']=False;b=seal_predictions(b);assert evaluate_quality(s,b,refs)['status']=='incomplete';return
    with pytest.raises(ValueError):evaluate_quality(s,b,refs)


@pytest.mark.parametrize('role',['s2','safety','monitor'])
def test_once_per_observation_transport_and_blind_preview(tmp_path,role):
    root,s,refs,_=quality_fixture(tmp_path,role)
    if role=='monitor':
        c=config(role);c['binding']['kwargs']['async_concurrency']=2;s=prepare_quality(c,s['corpus'],root)
    async def run():return await run_quality(s,root,Transport(probe_plan(s['config'])),tmp_path/'run')
    b=asyncio.run(run());assert b['status']=='complete' and b['drained'] and b['actions_executed']==0
    assert len(b['rows'])==48 and all(x['status']=='ok' for x in b['rows'])
    assert {x['sample_id'] for x in b['rows']}=={x['sample_id'] for x in s['samples']}
    if role=='safety':
        export_review_images(s,root,tmp_path/'review');assert len(list((tmp_path/'review').glob('*.png')))==48
        assert 'simulator_success' not in json.dumps(annotation_template(s))


def test_quality_timeout_keeps_late_result_and_unattempted_out_of_success(tmp_path):
    root,s,_,_=quality_fixture(tmp_path);c=config(request_timeout_s=.005,drain_timeout_s=.2);c['binding']['kwargs']['delay_s']=.03
    s=prepare_quality(c,s['corpus'],root)
    async def run():return await run_quality(s,root,Transport(probe_plan(c)),tmp_path/'timeout')
    b=asyncio.run(run());assert b['status']=='partial' and b['drained']
    assert b['rows'][0]['status']=='timeout' and all(x['status']=='not_run' for x in b['rows'][1:])
    assert evaluate_quality(s,b)['status']=='incomplete'
