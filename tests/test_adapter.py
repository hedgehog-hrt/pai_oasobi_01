from dataclasses import asdict, replace
import itertools
import threading

import numpy as np
import pytest

from rosa_repro.adapter import (BatchOneWorker, MeasuredClient, ProtocolError, Request,
                               check_action_batch_one, JsonlSink)
from rosa_repro.contracts import validate, MEASUREMENT_SCHEMA
from rosa_repro.measurement import evaluate_p1
from rosa_repro.regression import save_observation, load_observation, compare_actions


def observation(value=1):
    return {'video.image': np.zeros((1,1,2,2,3), np.uint8),
            'state.x': np.full((1,1,1), value, np.float32), 'task': ('pick up soup',)}


class Policy:
    def __init__(self):
        self.calls = []
    def get_action(self, obs):
        self.calls.append(obs)
        return {'action.x': np.repeat(obs['state.x'], 8, axis=1)}, {}


def worker():
    return BatchOneWorker(Policy(), stateless_verified=True)


def request(w, robot='r0', episode='e0', request_id='q0', seq=0):
    ack = w.reset('run', robot, episode)
    return Request('run', robot, episode, request_id, 'attempt', 's1', seq, ack['generation'])


def client(w, records, **kwargs):
    return MeasuredClient(run_id='run', robot_id='r0', reset_call=w.reset, predict_call=w.predict,
                          sink=records.append, clock_domain='test-clock',
                          clock=iter(range(1000,100000,100)).__next__, **kwargs)


def test_public_policy_api_receives_unchanged_observation_and_preserves_output():
    w=worker(); req=request(w); obs=observation()
    r=w.predict(asdict(req),obs)
    assert w.policy.calls == [obs] and w.policy.calls[0] is obs
    assert r['metadata']==asdict(req) and r['actual_batch_size']==1
    assert r['action']['action.x'].shape==(1,8,1)


def test_stateful_backend_requires_explicit_review():
    with pytest.raises(ProtocolError): BatchOneWorker(Policy())


def test_robot_identity_and_reset_isolation():
    w=worker(); a=request(w); b=request(w,'r1',request_id='b')
    w.reset('run','r0','e1')
    with pytest.raises(ProtocolError,match='Stale'): w.predict(asdict(a),observation())
    assert w.predict(asdict(b),observation(2))['action']['action.x'][0,0,0]==2


def test_duplicate_episode_request_and_out_of_order_sequence():
    w=worker(); req=request(w)
    with pytest.raises(ProtocolError,match='reused'): w.reset('run','r0','e0')
    w.predict(asdict(req),observation())
    with pytest.raises(ProtocolError,match='Duplicate'): w.predict(asdict(req),observation())
    with pytest.raises(ProtocolError,match='out-of-order'):
        w.predict(asdict(replace(req,request_id='q1')),observation())


@pytest.mark.parametrize('bad', [np.full((1,8,1),np.nan,np.float32),
                                 np.ones((2,8,1),np.float32),np.ones((1,8,1),np.float64),
                                 np.ones((1,0,1),np.float32)])
def test_invalid_actions_are_refused(bad):
    with pytest.raises(ProtocolError): check_action_batch_one({'action.x':bad})


def test_batch_greater_than_one_and_nonfinite_observations_refused():
    w=worker(); req=request(w)
    obs=observation(); obs['state.x']=np.ones((2,1,1),np.float32)
    with pytest.raises(ProtocolError): w.predict(asdict(req),obs)
    obs=observation(); obs['state.x'][0,0,0]=np.inf
    with pytest.raises(ProtocolError): w.predict(asdict(req),obs)
    assert not w.policy.calls


def test_reset_during_model_call_cancels_response():
    w=worker(); req=request(w); entered=threading.Event(); release=threading.Event()
    original=w.policy.get_action
    def blocked(obs):
        entered.set(); assert release.wait(3); return original(obs)
    w.policy.get_action=blocked
    result=[]
    t=threading.Thread(target=lambda: result.append(w.predict(asdict(req),observation())))
    t.start(); assert entered.wait(3)
    w.reset('run','r0','e1'); release.set(); t.join(3)
    assert not t.is_alive() and result[0]['status']=='cancelled' and result[0]['action'] is None


def test_same_robot_concurrent_request_refused_and_model_is_shared_serially():
    w=worker(); a=request(w); b=request(w,'r1',request_id='b')
    entered=threading.Event(); release=threading.Event(); original=w.policy.get_action
    active=0; maximum=0
    def blocked(obs):
        nonlocal active,maximum
        active+=1; maximum=max(maximum,active); entered.set()
        assert release.wait(3)
        result=original(obs); active-=1; return result
    w.policy.get_action=blocked; results=[]
    t=threading.Thread(target=lambda: results.append(w.predict(asdict(a),observation())))
    t.start(); assert entered.wait(3)
    with pytest.raises(ProtocolError,match='Concurrent'):
        w.predict(asdict(replace(a,request_id='new',observation_seq=1)),observation())
    u=threading.Thread(target=lambda: results.append(w.predict(asdict(b),observation(2))))
    u.start(); release.set(); t.join(3); u.join(3)
    assert len(results)==2 and maximum==1


def test_request_and_actual_action_records_satisfy_r00_and_evaluator():
    records=[]; c=client(worker(),records); c.reset('e0')
    q=c.predict(observation())
    assert c.apply(q,lambda action: 5)==5
    with pytest.raises(ProtocolError): c.apply(q,lambda _:8)
    for r in records: validate(r,MEASUREMENT_SCHEMA)
    result=evaluate_p1(records,run_id='run',clock_domain='test-clock',window_start_ns=0,window_end_ns=10000)
    assert result['requests']==1 and result['raw_action_chunks']==1 and result['qualified_action_chunks']==1
    assert records[1]['executed_steps']==5


def test_client_pending_response_cannot_survive_reset():
    records=[]; c=client(worker(),records); c.reset('e0'); q=c.predict(observation())
    c.reset('e1')
    with pytest.raises(ProtocolError): c.apply(q,lambda _:8)
    assert len(records)==1


def test_client_disallows_next_request_until_action_consumed():
    c=client(worker(),[]); c.reset('e0'); q=c.predict(observation())
    with pytest.raises(ProtocolError): c.predict(observation())
    c.apply(q,lambda _:0)
    assert c.predict(observation())


@pytest.mark.parametrize('corruption',['robot_id','episode_id','request_id','observation_seq','generation'])
def test_response_mismatch_is_never_applied(corruption):
    records=[]; w=worker(); c=client(w,records); c.reset('e0')
    def corrupt(meta,obs):
        r=w.predict(meta,obs); r['metadata'][corruption]='wrong'; return r
    c.predict_call=corrupt
    with pytest.raises(ProtocolError,match='identity'): c.predict(observation())
    assert records[0]['status']=='error' and not records[0]['valid_output'] and not c.pending


def test_timeout_stays_in_measurement_denominator():
    records=[]; c=client(worker(),records); c.reset('e0')
    def fail(*args): raise TimeoutError('transport timeout')
    c.predict_call=fail
    with pytest.raises(TimeoutError): c.predict(observation())
    r=evaluate_p1(records,run_id='run',clock_domain='test-clock',window_start_ns=0,window_end_ns=10000)
    assert r['requests']==1 and r['deadline_unqualified_requests']==1 and r['raw_action_chunks']==0


def test_executor_failure_never_fabricates_completed_action():
    records=[]; c=client(worker(),records); c.reset('e0'); q=c.predict(observation())
    def fail(action): raise RuntimeError('execution stopped; partial steps unknown')
    with pytest.raises(RuntimeError): c.apply(q,fail)
    assert len(records)==1
    with pytest.raises(ProtocolError): c.apply(q,lambda _:8)


def test_nonfinite_model_output_is_recorded_error():
    records=[]; w=worker(); w.policy.get_action=lambda _:({'action.x':np.full((1,8,1),np.nan,np.float32)}, {})
    c=client(w,records); c.reset('e0')
    with pytest.raises(ProtocolError): c.predict(observation())
    assert records[0]['status']=='error' and not c.pending


def test_saved_observation_exact_roundtrip_and_tamper_detection(tmp_path):
    obs=observation(); path=tmp_path/'case'
    save_observation(path,obs,{'initial_state_id':3})
    restored, manifest=load_observation(path)
    assert manifest['provenance']['initial_state_id']==3
    assert restored['task']==obs['task']
    for k in ('state.x','video.image'):
        assert np.array_equal(restored[k],obs[k]) and restored[k].dtype==obs[k].dtype
    with (path/'arrays.npz').open('ab') as f: f.write(b'tamper')
    with pytest.raises(ValueError,match='hash'): load_observation(path)


def test_comparison_reports_real_numeric_error_and_rejects_broadcast():
    a={'action.x':np.zeros((1,8,1),np.float32)}
    b={'action.x':np.full((1,8,1),0.01,np.float32)}
    assert not compare_actions(a,b)['passed']
    assert compare_actions(a,b,atol=.011)['passed']
    b={'action.x':np.zeros((1,8,2),np.float32)}
    assert not compare_actions(a,b)['passed']


def test_jsonl_sink_exclusive_and_json_valid(tmp_path):
    import json
    p=tmp_path/'trace.jsonl'; sink=JsonlSink(p); sink({'x':1}); sink.close()
    assert json.loads(p.read_text())=={'x':1}
    with pytest.raises(FileExistsError): JsonlSink(p)


def test_log_failure_revokes_execution_token():
    c=client(worker(),[]); c.reset('e0')
    def fail(record): raise OSError('disk full')
    c.sink=fail
    with pytest.raises(OSError): c.predict(observation())
    assert not c.pending and not c.busy


def test_client_reset_while_transport_inflight_discards_even_valid_old_response():
    records=[]; w=worker(); c=client(w,records); c.reset('e0')
    entered=threading.Event(); release=threading.Event()
    def delayed(meta,obs):
        response=w.predict(meta,obs)
        entered.set(); assert release.wait(3)
        return response
    c.predict_call=delayed
    result=[]
    t=threading.Thread(target=lambda:result.append(c.predict(observation())))
    t.start(); assert entered.wait(3); c.reset('e1'); release.set(); t.join(3)
    assert result==[None] and records[0]['status']=='cancelled' and not c.pending


@pytest.mark.parametrize('size',[None,True,2])
def test_invalid_batch_evidence_is_rejected(size):
    records=[]; w=worker(); c=client(w,records); c.reset('e0')
    def corrupt(meta,obs):
        response=w.predict(meta,obs); response['actual_batch_size']=size; return response
    c.predict_call=corrupt
    with pytest.raises(ProtocolError): c.predict(observation())
    assert records[0]['status']=='error' and not c.pending


def test_warmup_actions_do_not_become_measured_actions():
    records=[]; c=client(worker(),records); c.reset('e0')
    q=c.predict(observation(),warmup=True); c.apply(q,lambda _:8)
    r=evaluate_p1(records,run_id='run',clock_domain='test-clock',window_start_ns=0,window_end_ns=10000)
    assert r['requests']==0 and r['raw_action_chunks']==0


def test_checkpoint_guard_refuses_wrong_baseline(tmp_path):
    from rosa_repro.regression import verify_checkpoint_config
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text('{}')
    with pytest.raises(ValueError,match='differs'): verify_checkpoint_config(tmp_path)
