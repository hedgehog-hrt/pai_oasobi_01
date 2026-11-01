import json
from copy import deepcopy
from itertools import product
import pytest
from rosa_repro.hybrid_safety import config,monitored_pair,valid_preview,valid_receipt,audit_guard,PROTOCOL
from rosa_repro.paired import digest_value
from rosa_repro.vlm_contract import parse_generation,default_config,validate_config,generation_schema,runtime_fingerprint

@pytest.mark.parametrize('visibility,person,collision',product(('clear','unknown'),('present','absent','unknown'),('present','absent','unknown')))
def test_visual_unknown_is_not_silently_accepted(visibility,person,collision):
    labels=dict(visibility=visibility,person=person,collision=collision)
    r=parse_generation('safety',json.dumps(labels),'stop','safety_visual_guard_v1')
    assert r['stop']==(visibility!='clear' or person!='absent' or collision=='present')
    # Collision uncertainty delegates only this check to the required geometry guard.

@pytest.mark.parametrize('raw,finish',[('{"visibility":"clear","person":"absent","collision":"unknown"}','length'),
    ('{"visibility":"clear","person":"absent"}','stop'),
    ('{"visibility":"clear","person":"absent","collision":"unknown","extra":"safe"}','stop'),
    ('{"visibility":"clear","person":"absent","collision":"bad"}','stop')])
def test_malformed_or_truncated_visual_output_never_clears(raw,finish):
    with pytest.raises(ValueError):parse_generation('safety',raw,finish,'safety_visual_guard_v1')

@pytest.mark.parametrize('a,b,kwargs,check',[
 ('gripper0_finger1_collision','alphabet_soup_1_g1',{},False),
 ('alphabet_soup_1_g1','basket_1_g1',{},False),
 ('alphabet_soup_1_g1','floor',{},False),
 ('robot0_link0_collision','floor',{},False),
 ('robot0_link5_collision','floor',{},True),
 ('gripper0_hand_collision','basket_1_g1',{},True),
 ('robot0_link5_collision','alphabet_soup_1_g1',{},True),
 ('alphabet_soup_1_g1','milk_1_g1',{},True),
 ('robot0_link2_collision','robot0_link3_collision',{'adjacent':True},False),
 ('robot0_link2_collision','robot0_link6_collision',{},True),
 ('milk_1_g1','floor',{},False)])
def test_contact_policy(a,b,kwargs,check):
    assert monitored_pair(a,b,**kwargs)==check
    assert monitored_pair(b,a,**kwargs)==check


def fixture():
    cfg=config();h='a'*64
    p=dict(protocol=PROTOCOL,config_sha256=digest_value(cfg),action_sha256=h,requested_steps=8,
        live_steps_executed=0,status='clear',timely=True,elapsed_ms=100.,recovery_ms=0.,physics_preview_ms=100.,lease_id='b'*32,
        predicted_steps=8,checked_physics_substeps=200,physics_timestep_s=.002,monitored_geom_pairs=10,
        state_before_sha256='c'*64,predicted_state_after_sha256='d'*64)
    receipt=dict(lease_id=p['lease_id'],action_sha256=h,state_after_sha256='d'*64,twin_match=True)
    return cfg,p,receipt

@pytest.mark.parametrize('key,value', [('status','unknown'),('status','blocked'),('timely',False),('elapsed_ms',2501.),('physics_preview_ms',501.),('recovery_ms',2001.),
    ('elapsed_ms',float('nan')),('action_sha256','z'*64),('requested_steps',7),('live_steps_executed',1),
    ('config_sha256','z'*64),('checked_physics_substeps',0),('lease_id',None)])
def test_invalid_preview_fails_closed(key,value):
    cfg,p,r=fixture();assert valid_preview(p,cfg,'a'*64,8) and valid_receipt(r,p,8)
    p[key]=value;assert not valid_preview(p,cfg,'a'*64,8)


def audit_fixture():
    cfg,p,receipt=fixture()
    comp=dict(id='safety',role='safety',period_ns=500000000,deadline_ms=500.)
    plan=dict(collision_guard=cfg,classes=[dict(audit=dict(components=[comp]),robots=[dict(robot_id='r')])])
    labels=dict(visibility='clear',person='absent',collision='unknown');text=json.dumps(labels)
    parsed=parse_generation('safety',text,'stop','safety_visual_guard_v1')
    records=[dict(record_type='request',request_id='visual',robot_id='r',episode_id='e',component_id='safety',status='ok',valid_output=True,scheduled_ns=0,finished_ns=100),
        dict(record_type='request',request_id='s1',robot_id='r',episode_id='e',component_id='s1',finished_ns=150),
        dict(record_type='action',request_id='s1',robot_id='r',episode_id='e',start_ns=300,applied=True,executed_steps=8,safety_stopped=False)]
    intents=[dict(kind='hybrid_safety_gate',request_id='s1',checked_at_ns=300,visual_request_id='visual',cause='clear',stopped=False),
        dict(kind='collision_guard_preview',request_id='s1',start_ns=160,end_ns=250,evidence=p),
        dict(kind='collision_guard_applied',request_id='s1',receipt=receipt),dict(kind='action_tensor',request_id='s1',requested_steps=8)]
    details=dict(visual=dict(semantic_output=parsed,output_sha256=digest_value(parsed),backend_evidence=dict(raw_generation=text,finish_reason='stop')),s1=dict(output_sha256='a'*64))
    return plan,records,intents,details

@pytest.mark.parametrize('tamper',['missing','changed_action','expired','not_latest','receipt','late_preview','duplicate','unsafe'])
def test_raw_audit_requires_both_authorizations(tamper):
    p,r,i,d=audit_fixture();assert audit_guard(p,r,i,d)['applied_guard_leases']==1
    if tamper=='missing':i.pop(1)
    elif tamper=='changed_action':d['s1']['output_sha256']='z'*64
    elif tamper=='expired':r[-1]['start_ns']=1000000000;i[0]['checked_at_ns']=1000000000
    elif tamper=='not_latest':r.insert(0,dict(r[0],request_id='new',finished_ns=200))
    elif tamper=='receipt':i[2]['receipt']['state_after_sha256']='z'*64
    elif tamper=='late_preview':i[1]['end_ns']=301
    elif tamper=='duplicate':i.append(deepcopy(i[1]))
    elif tamper=='unsafe':d['visual']['semantic_output']['stop']=True
    with pytest.raises(ValueError):audit_guard(p,r,i,d)


def test_guard_format_cannot_be_used_without_geometry():
    from rosa_repro.prototype_plan import prepare
    from rosa_repro.profile_protocol import template
    values={r:template(r,'groot' if r=='s1' else 'vllm') for r in ('s1','safety','monitor')}
    for role,c in values.items():
        c['hardware_kind']='GPU';c['binding']['runtime_env']={'py_executable':'/opt/python'}
        c['binding']['kwargs']['model_path']='/models/'+role
        if role=='s1':c['binding']['kwargs'].update(repo='/repo',stateless_verified=True);c['runtime_fingerprint']='numeric-fixture'
        else:
            conf=c['binding']['kwargs']['config'];conf['gpu_memory_utilization']=.2
            if role=='safety':conf['structured_output']='safety_visual_guard_v1'
            c['runtime_fingerprint']=runtime_fingerprint(conf)
    with pytest.raises(ValueError,match='paired'):prepare(values,{'P3':1})
    plan=prepare(values,{'P3':1},collision_guard=config())
    assert plan['collision_guard']['simulation_only']


def test_geometry_queries_do_not_mutate_forecast_caches():
    import numpy as np
    from types import SimpleNamespace
    from rosa_repro.hybrid_safety import GeometryChecker
    class NativeDataFixture:
        def __init__(self):
            self.cache_updates=0;self.contact=[];self.ncon=0;self.geom_xpos=np.array([[0.,0.,0.],[10.,0.,0.]])
        def __copy__(self):return NativeDataFixture()
    def refresh(model,data):data.cache_updates+=1
    checker=GeometryChecker.__new__(GeometryChecker)
    checker.source_data=NativeDataFixture();checker.data=None;checker.model=object()
    checker.mujoco=SimpleNamespace(mj_fwdPosition=refresh)
    checker.pairs=np.array([[0,1]],dtype=np.int32);checker.planes=np.array([False,False]);checker.radii=np.array([.1,.1]);checker.margin=.002
    checker.check();checker.check()
    assert checker.source_data.cache_updates==0
    assert checker.data is not checker.source_data and checker.data.cache_updates==1
