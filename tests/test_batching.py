from dataclasses import asdict
import numpy as np
import pytest
from rosa_repro.adapter import Request,ProtocolError,MeasuredClient
from rosa_repro.batching import combine_observations,split_actions,SessionBook
from rosa_repro.dummy import observation,DummyPolicy


def test_true_tensor_batch_and_identity_preserving_split():
    inputs=[observation(i) for i in range(4)]
    merged=combine_observations(inputs)
    assert merged['video.image'].shape==(4,1,2,2,3) and merged['task']==['CPU fixture']*4
    action,_=DummyPolicy().get_action(merged)
    outputs=split_actions(action,4)
    assert [float(o['action.x'][0,0,0]) for o in outputs]==[0,1,2,3]


def test_mixed_shapes_refused():
    a=observation(1);b=observation(2);b['video.image']=np.zeros((1,1,3,3,3),np.uint8)
    with pytest.raises(ProtocolError):combine_observations([a,b])


def test_wrong_model_batch_or_nonfinite_output_refused():
    with pytest.raises(ProtocolError):split_actions({'action.x':np.zeros((2,8,1),np.float32)},3)
    with pytest.raises(ProtocolError):split_actions({'action.x':np.full((2,8,1),np.nan,np.float32)},2)


def test_reset_invalidates_admitted_ticket_and_does_not_release_new_busy_state():
    b=SessionBook();ack=b.reset('run','r0','e0')
    req=Request('run','r0','e0','q','a','s1',0,ack['generation'])
    old=b.accept(asdict(req),observation(1));ack=b.reset('run','r0','e1')
    req=Request('run','r0','e1','q1','a','s1',0,ack['generation'])
    new=b.accept(asdict(req),observation(2));b.finish(old)
    assert not b.current(old) and b.current(new) and new['state']['busy']


def test_measured_client_accepts_declared_multibatch_and_keeps_batch_evidence():
    records=[]
    def reset(run_id,robot_id,episode_id):return dict(run_id=run_id,robot_id=robot_id,episode_id=episode_id,generation=0)
    def predict(metadata,obs):
        return dict(metadata=metadata,status='ok',action=DummyPolicy().get_action(obs)[0],batch_id='b',actual_batch_size=4)
    c=MeasuredClient(run_id='run',robot_id='r',reset_call=reset,predict_call=predict,sink=records.append,clock_domain='clock',max_batch_size=4)
    c.reset('e');q=c.predict(observation(1));c.apply(q,lambda _:8)
    assert records[0]['actual_batch_size']==4 and records[1]['executed_steps']==8
