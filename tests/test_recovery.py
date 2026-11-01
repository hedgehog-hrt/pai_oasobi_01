import asyncio,json,time
from copy import deepcopy
import numpy as np
import pytest
from test_hybrid_safety import fixture
from rosa_repro.recovery_policy import config,candidates,Limits,valid_authorization
from rosa_repro.recovery_runtime import attempt_retreat
from rosa_repro.recovery_audit import audit_recovery
from rosa_repro.paired import digest_value
from rosa_repro.regression import sha256
from rosa_repro.vlm_contract import parse_generation


def template():return {'action.'+k:np.zeros((1,16,1),dtype=np.float32) for k in ('x','y','z','roll','pitch','yaw','gripper')}

def proof(action,source='retreat'):
    guard,e,_=fixture();policy=config();e.update(action_sha256=digest_value(action),motion_source=source,
        recovery_policy_sha256=digest_value(policy),checkpoint_protocol='libero_t01_native_shadow_checkpoint_v1',
        checkpoint_sha256='f'*64,nominal_geometry_status='clear',end_effector_delta_m=[0.,0.,.01],displacement_m=.01)
    _,escape=candidates(action,policy)[0];w=deepcopy(e)
    w.update(action_sha256=digest_value(escape),state_before_sha256=e['predicted_state_after_sha256'],lease_id='e'*32)
    e['terminal_witness']=dict(kind='retreat',candidate='up_0.1',evidence=w)
    return guard,e


def test_bounded_candidates_and_no_budget_reset_by_stopped_proposals():
    a=template();choices=candidates(a,config());assert len(choices)==3
    for _,c in choices:
        assert np.all(c['action.gripper']==.5) and np.all(c['action.x']==0)
    l=Limits(config())
    assert not l.accepts([0,0,-.01]) and not l.accepts([0,0,.081])
    l.applied('retreat',.01);l.applied('retreat',.01);assert not l.available()
    with pytest.raises(ValueError):l.applied('retreat',.01)
    l.applied('s1');l.applied('s1');assert not l.available()
    l.applied('s1');assert l.available()
    for _ in range(4):
        l.applied('retreat',.01)
        for _ in range(3):l.applied('s1')
    assert not l.available()

@pytest.mark.parametrize('bad',['policy','source','state','witness','direction','distance','late'])
def test_recovery_proof_rejects_changed_or_unverified_witness(bad):
    a=template();g,e=proof(a);assert valid_authorization(e,g,config(),a,8,'retreat')
    if bad=='policy':e['recovery_policy_sha256']='0'*64
    elif bad=='source':e['motion_source']='s1'
    elif bad=='state':e['terminal_witness']['evidence']['state_before_sha256']='0'*64
    elif bad=='witness':e['terminal_witness']=None
    elif bad=='direction':e['terminal_witness']['evidence']['end_effector_delta_m']=[0,0,-.01]
    elif bad=='distance':e['terminal_witness']['evidence']['end_effector_delta_m']=[0,0,.1]
    elif bad=='late':e['elapsed_ms']=2501
    assert not valid_authorization(e,g,config(),a,8,'retreat')


def runtime_fixture(tmp_path,*,fresh=True,status='clear'):
    a=template();name,c=candidates(a,config())[0];g,e=proof(c);now=time.monotonic_ns()
    state=dict(observation_seq=1,safety_observation_version=0 if fresh else -1,safety_request_id='visual',plan=('old','old'),safety=False)
    outcome=dict(steps_unknown=False,actual_steps=0,action_chunks=0);intents=[]
    class Sim:
        calls=0
        async def retreat(self,a,n):return dict(status=status,reason=status,candidate=name,action=c,evidence=e,candidates=[],limits=Limits(config()).evidence())
        async def step(self,a,n):
            self.calls+=1;l=Limits(config());l.applied('retreat',.01)
            return dict(steps=8,success=False,done=False,observation={},collision_guard=dict(lease_id=e['lease_id'],action_sha256=e['action_sha256'],state_after_sha256=e['predicted_state_after_sha256'],twin_match=True),
                recovery_state=dict(policy_sha256=digest_value(config()),motion_source='retreat',**l.evidence()))
    sim=Sim();plan=dict(collision_guard=g,recovery_policy=config(),classes=[dict(audit=dict(id='P4',components=[dict(id='safety',role='safety',period_ns=500000000,deadline_ms=500)]),robots=[dict(robot_id='r')])])
    def snapshot(obs):state['observation_seq']+=1
    async def until(_):pass
    parent=dict(request_id='s1',robot_id='r',episode_id='e')
    result=asyncio.run(attempt_retreat(sim=sim,plan=plan,state=state,outcome=outcome,parent=parent,template=a,limit=8,end=now+10**9,out=tmp_path,
        life=intents.append,snapshot=snapshot,capture_size=lambda _:None,visual_blocked=lambda:False,until=until,action_time_s=.4))
    return locals()

@pytest.mark.parametrize('fresh,status,applied',[(True,'clear',True),(False,'clear',False),(True,'unknown',False),(True,'blocked',False)])
def test_recovery_requires_fresh_visual_and_clears_plan_and_safety(tmp_path,fresh,status,applied):
    f=runtime_fixture(tmp_path,fresh=fresh,status=status);e=f['intents'][-1]
    assert e['applied']==applied and f['sim'].calls==int(applied)
    if applied:
        assert f['state']['plan'] is None and f['state']['safety'] is None
        assert f['state']['minimum_safety_observation_version']==1
        assert f['outcome']['actual_steps']==8 and f['outcome']['action_chunks']==1
        assert e['counts_as_S1_success'] is False


def audit_fixture(tmp_path):
    f=runtime_fixture(tmp_path);i=f['intents'];e=i[-1];now=f['now'];a=f['a']
    file=tmp_path/'original.npz';np.savez_compressed(file,**a)
    i.extend([dict(kind='action_tensor',request_id='s1',file=file.name,sha256=sha256(file),requested_steps=8),
        dict(kind='collision_guard_preview',request_id='s1',evidence={'status':'blocked'})])
    r=[dict(record_type='request',request_id='visual',robot_id='r',episode_id='e',component_id='safety',status='ok',valid_output=True,scheduled_ns=now-100,finished_ns=now),
        dict(record_type='action',request_id='s1',robot_id='r',episode_id='e',start_ns=now,end_ns=now,applied=False,safety_stopped=True,executed_steps=0)]
    text=json.dumps(dict(visibility='clear',person='absent',collision='unknown'));parsed=parse_generation('safety',text,'stop','safety_visual_guard_v1')
    r.append(dict(record_type='request',request_id='s1',robot_id='r',episode_id='e',component_id='s1',warmup=False,sent_ns=now,finished_ns=now))
    d={'visual':dict(observation_version=0,semantic_output=parsed,output_sha256=digest_value(parsed),backend_evidence=dict(raw_generation=text,finish_reason='stop'))}
    return f['plan'],r,i,d,now

@pytest.mark.parametrize('bad',['none','stale_visual','receipt','source','unknown_parent','counter','launch','candidate','false_resume'])
def test_reaudit_preserves_original_refusal_and_rejects_forged_recovery(tmp_path,bad):
    p,r,i,d,t=audit_fixture(tmp_path);e=next(x for x in i if x['kind']=='recovery_decision')
    if bad=='stale_visual':d['visual']['observation_version']=-1
    elif bad=='receipt':e['receipt']['state_after_sha256']='0'*64
    elif bad=='source':e['counts_as_S1_success']=True
    elif bad=='unknown_parent':i[-1]['evidence']['status']='unknown'
    elif bad=='counter':e['recovery_state']['retreats']=2
    elif bad=='launch':i[:]=[x for x in i if x['kind']!='recovery_action_started']
    elif bad=='candidate':e['response']['candidate']='up_0.5'
    elif bad=='false_resume':i.append(dict(kind='recovery_s1_resumed',recovery_id=e['recovery_id'],request_id='s1'))
    if bad!='none':
        with pytest.raises(ValueError):audit_recovery(tmp_path,p,r,i,d,t-1,t+10**9)
    else:
        metrics,applied=audit_recovery(tmp_path,p,r,i,d,t-1,t+10**9);m=metrics['classes']['P4']
        assert m['original_s1_refusal_rate']==1 and m['final_stop_rate']==0 and m['sustained_s1_resumptions']==0
        assert len(applied)==1 and m['original_s1_applied']==0

@pytest.mark.parametrize('bad',['none','stale_s1','stale_safety','old_plan'])
def test_recovery_resumption_requires_new_inputs_and_p4_replanning(tmp_path,bad):
    p,r,i,d,t=audit_fixture(tmp_path);retreat=next(x for x in i if x['kind']=='recovery_decision')
    at=retreat['end_ns']+100;last_visual=deepcopy(d['visual']);last_visual['observation_version']=1
    d['fresh']=last_visual
    r.append(dict(r[0],request_id='fresh',scheduled_ns=at-20,finished_ns=at-10))
    r.append(dict(record_type='request',request_id='new-plan',sent_ns=at-5,robot_id='r',episode_id='e',component_id='s2',finished_ns=at-1))
    d['new-plan']=dict(observation_version=1)
    for n in range(3):
        a=template();g,e=proof(a,'s1');e['lease_id']=str(n)*32;rid='resumed-'+str(n);path=tmp_path/(rid+'.npz');np.savez_compressed(path,**a)
        r.append(dict(record_type='action',request_id=rid,robot_id='r',episode_id='e',start_ns=at+n*100,applied=True,safety_stopped=False,executed_steps=8))
        i.extend([dict(kind='action_tensor',request_id=rid,file=path.name,sha256=sha256(path),requested_steps=8),
            dict(kind='collision_guard_preview',request_id=rid,evidence=e),dict(kind='hybrid_safety_gate',request_id=rid,visual_request_id='fresh')])
        r.append(dict(record_type='request',request_id=rid,robot_id='r',episode_id='e',component_id='s1',warmup=False,sent_ns=at+n*100,finished_ns=at+n*100))
        d[rid]=dict(observation_version=1,plan_request_id='new-plan')
    i.append(dict(kind='recovery_s1_resumed',recovery_id=retreat['recovery_id'],request_id='resumed-2'))
    if bad=='stale_s1':d['resumed-0']['observation_version']=0
    elif bad=='stale_safety':d['fresh']['observation_version']=0
    elif bad=='old_plan':d['new-plan']['observation_version']=0
    if bad!='none':
        with pytest.raises(ValueError):audit_recovery(tmp_path,p,r,i,d,t-1,t+10**9)
    else:
        m,_=audit_recovery(tmp_path,p,r,i,d,t-1,t+10**9)
        assert m['classes']['P4']['sustained_s1_resumptions']==1


def test_recovery_denominator_excludes_warmup_requests_crossing_boundary(tmp_path):
    p,r,i,d,t=audit_fixture(tmp_path)
    parent=next(x for x in r if x['record_type']=='request' and x['request_id']=='s1')
    parent.update(warmup=True,sent_ns=t-100)
    m,_=audit_recovery(tmp_path,p,r,i,d,t-1,t+10**9)
    assert m['classes']['P4']['original_s1_decisions']==0 and m['classes']['P4']['retreats_applied']==0
