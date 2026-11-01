from copy import deepcopy
import pytest
from rosa_repro.multicomponent import evaluate_multicomponent
from rosa_repro.measurement import evaluate_p1
from multi_helpers import fixture,req,action,NS


@pytest.mark.parametrize('pipeline',['P1','P2','P3','P4'])
def test_all_declared_components_qualify(pipeline):
    r=evaluate_multicomponent(fixture(pipeline))
    assert r['qualified_action_chunks']==1 and r['weighted_Q']==.5
    assert r['classes']['c0']['raw_action_chunks']==1


def test_p1_reduction_matches_previous_evaluator():
    d=fixture('P1',3);old=[]
    for record in d['records']:
        r=dict(record,schema_version='1.0')
        for k in ('slot_index','plan_request_id','safety_stopped'):r.pop(k,None)
        old.append(r)
    a=evaluate_p1(old,run_id='run',clock_domain='clock',window_start_ns=2*NS,window_end_ns=4*NS)
    b=evaluate_multicomponent(d)
    assert a['qualified_action_chunks']==b['qualified_action_chunks']
    assert a['qualified_action_rate_hz']==b['weighted_Q']


@pytest.mark.parametrize('component,pipeline',[('monitor','P2'),('safety','P3'),('monitor','P4')])
def test_missing_periodic_slot_is_generated_and_disqualifies(component,pipeline):
    d=fixture(pipeline);d['records']=[r for r in d['records'] if r['component_id']!=component]
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==0
    assert component+'_slot_missing' in result['actions'][0]['reasons']
    assert result['periodic_slot_metrics']['c0'][component]['status_counts']['missing']>0


def test_late_send_cannot_hide_periodic_deadline_violation():
    d=fixture('P2');d['records'][0]=req('late','monitor',1.99,2.04,scheduled=0,slot=0)
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==0
    metrics=result['component_request_metrics']['c0']['monitor']
    assert result['slots'][0]['scheduled_to_finished_ms']==[2040]
    assert result['slots'][0]['status']=='violated'


def test_retry_within_original_slot_deadline_can_qualify_but_not_inflate_slot_count():
    d=fixture('P2');d['records'][0]=req('failed','monitor',0,1,status='timeout',valid=False,slot=0)
    d['records'].append(req('retry','monitor',1.8,1.9,scheduled=0,slot=0))
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==1 and result['slots'][0]['attempts']==2
    assert result['periodic_slot_metrics']['c0']['monitor']['expected_matured_slots']==1


def test_exact_deadline_inclusive_but_future_success_cannot_rescue_expired_slot():
    d=fixture('P2');d['window_start_ns']=1_900_000_000
    d['records']=[req('monitor','monitor',0,2,slot=0)]
    r=req('s1','s1',1.95,2);d['records']+=[r,action(r)]
    assert evaluate_multicomponent(d)['qualified_action_chunks']==1
    d['records'][0]['finished_ns']+=1
    assert evaluate_multicomponent(d)['qualified_action_chunks']==0


def test_startup_and_pending_slots_do_not_create_qualified_actions():
    d=fixture('P2');d['window_start_ns']=0
    r=req('early','s1',.1);d['records']+=[r,action(r)]
    result=evaluate_multicomponent(d)
    assert result['classes']['c0']['startup_robot_seconds']==2
    assert result['qualified_action_chunks']==1
    assert any('monitor_startup_pending' in a['reasons'] for a in result['actions'])


def test_plan_horizon_counts_s1_invocations_including_unapplied_requests():
    d=fixture('P4',11)
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==10
    assert result['actions'][-1]['plan_use_index']==11
    assert 's2_horizon_exceeded' in result['actions'][-1]['reasons']
    # Removing an action does not refund a plan use.
    d['records']=[r for r in d['records'] if r.get('action_id')!='act-s1-0']
    assert evaluate_multicomponent(d)['qualified_action_chunks']==9


@pytest.mark.parametrize('mode',['missing','late','not_ready','invalid'])
def test_s2_plan_failures(mode):
    d=fixture('P4')
    p=next(r for r in d['records'] if r.get('request_id')=='plan')
    if mode=='missing':d['records'].remove(p)
    elif mode=='late':p['finished_ns']=2_010_000_000
    elif mode=='not_ready':p['sent_ns']=1*NS;p['scheduled_ns']=1*NS;p['finished_ns']=2_010_000_000
    else:p.update(status='error',valid_output=False)
    r=evaluate_multicomponent(d)
    assert r['qualified_action_chunks']==0 and any(x.startswith('s2_') for x in r['actions'][0]['reasons'])


def test_stopped_action_separate_from_model_qualification():
    d=fixture('P3');a=d['records'][-1];a.update(safety_stopped=True,applied=False,executed_steps=0)
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==0 and result['actions'][0]['model_qualified']
    assert result['classes']['c0']['safety_stopped_decisions']==1
    a.update(applied=True,executed_steps=8)
    with pytest.raises(ValueError,match='stopped'):evaluate_multicomponent(d)


def test_p99_failure_does_not_zero_individual_qualified_actions():
    d=fixture('P1',2)
    r=next(r for r in d['records'] if r['request_id']=='s1-1' and r['record_type']=='request')
    r['finished_ns']=r['sent_ns']+201_000_000
    a=d['records'][-1];a['start_ns']=r['finished_ns'];a['end_ns']=r['finished_ns']+10_000_000
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==1
    assert not result['component_request_metrics']['c0']['s1']['send_p99_within_deadline']


def test_weighted_classes_and_episode_reset_isolation():
    a=fixture('P2');b=fixture('P4');b['classes'][0].update(id='c1',weight=3)
    b['episodes'][0].update(class_id='c1',robot_id='r1')
    for r in b['records']:
        r['robot_id']='r1';r['request_id']='b-'+r['request_id']
        if r['record_type']=='action':r['action_id']='b-'+r['action_id']
        else:
            if r['plan_request_id']:r['plan_request_id']='b-'+r['plan_request_id']
    a['classes']+=b['classes'];a['episodes']+=b['episodes'];a['records']+=b['records']
    result=evaluate_multicomponent(a)
    assert result['weighted_Q']==2
    # The second robot's requests cannot cite the first robot's plan (or S1).
    r=next(r for r in a['records'] if r.get('plan_request_id'))
    r['plan_request_id']='s1-0'
    with pytest.raises(ValueError,match='plan reference'):evaluate_multicomponent(a)


@pytest.mark.parametrize('change',['clock','slot','episode','duplicate','unfinalized'])
def test_corrupt_logs_rejected(change):
    d=fixture('P4')
    if change=='clock':d['records'][0]['clock_domain']='other'
    elif change=='slot':d['records'][0]['slot_index']=1
    elif change=='episode':d['records'][-1]['episode_id']='wrong'
    elif change=='duplicate':d['records'].append(deepcopy(d['records'][-1]))
    else:d['records'][0]['finished_ns']=11*NS
    with pytest.raises(ValueError):evaluate_multicomponent(d)


def test_random_periodic_oracle_uses_latest_matured_slot_not_latest_response():
    import random
    for seed in range(40):
        rng=random.Random(seed);d=fixture('P2');d['window_start_ns']=0;d['window_end_ns']=8*NS
        d['records']=[];outcomes={}
        for slot in range(4):
            state=rng.choice(['missing','good','late','timeout']);outcomes[slot]=state
            if state=='missing':continue
            sent=2*slot;finish=sent+(2.001 if state=='late' else .1)
            d['records'].append(req(f'm{slot}','monitor',sent,finish,slot=slot,
                                   status='timeout' if state=='timeout' else 'ok',valid=state!='timeout'))
        action_time=rng.uniform(2.01,7.9)
        r=req('x','s1',action_time-.05,action_time);d['records'] += [r,action(r)]
        latest=int((action_time-2)//2)
        expected=int(outcomes[latest]=='good')
        assert evaluate_multicomponent(d)['qualified_action_chunks']==expected


def test_periodic_state_does_not_carry_across_episode_reset():
    d=fixture('P2');d['episodes'][0]['end_ns']=2*NS
    d['episodes'].append(dict(class_id='c0',robot_id='r0',episode_id='e1',start_ns=2*NS,end_ns=10*NS))
    for r in d['records'][1:]:r['episode_id']='e1'
    result=evaluate_multicomponent(d)
    assert result['qualified_action_chunks']==0
    assert 'monitor_startup_pending' in result['actions'][0]['reasons']


def test_invalid_action_cannot_be_applied_even_when_other_components_qualify():
    d=fixture('P4');d['records'][-2].update(status='timeout',valid_output=False)
    with pytest.raises(ValueError,match='Invalid'):evaluate_multicomponent(d)
