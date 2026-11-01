import json,random
import numpy as np
import pytest,torch
from array_helpers import Client,capture,inputs
from rosa_repro.action_regression import collect_arrays,validate_run,seal_run,reaudit_case
from rosa_repro.dummy import DummyPolicy,observation
from rosa_repro.regression import sha256

def reseal(out):
    (out/'regression_run.json').unlink();seal_run(out)


@pytest.mark.parametrize('batch',[1,2,4])
def test_exact_arrays_offline_recomputed_for_serial_and_shared_batch(tmp_path,batch):
    assert capture(Client(),inputs(tmp_path/'in'),tmp_path/'out',batch_size=batch)=='passed'
    result=validate_run(tmp_path/'out',expected_batch_size=batch)
    assert result['raw_arrays_recomputed'] and result['cases']==4//batch
    assert not result['automatic_stochastic_batch_equivalence_claim']
    raw=tmp_path/'out/raw-case-0000'
    with np.load(raw/'reference-000.npz') as a,np.load(raw/'candidate-000.npz') as b:
        assert np.array_equal(a['action.x'],b['action.x'])
    assert not json.loads((tmp_path/'out/comparison_summary.json').read_text())['gpu_execution_confirmed_by_response']


def test_stochastic_batch_mismatch_preserves_arrays_and_default_rng_state(tmp_path):
    class Stochastic(DummyPolicy):
        def get_action(self,obs):
            actions,_=super().get_action(obs)
            actions['action.x']+=torch.rand(actions['action.x'].shape).numpy()+np.random.random()+random.random()
            return actions,{}
    policy=Stochastic();random.seed(101);np.random.seed(102);torch.manual_seed(103)
    py=random.getstate();npstate=np.random.get_state();ts=torch.random.get_rng_state().clone()
    assert collect_arrays(policy,[observation(1)],seed=42)['passed']
    assert collect_arrays(policy,[observation(1),observation(2)],seed=42)['passed']
    result=collect_arrays(policy,[observation(1),observation(2)],seed=42,comparison='serial_vs_batch')
    assert not result['passed'] and result['rng_alignment']=='unverified_across_batch_shapes'
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],npstate[1]) and torch.equal(torch.random.get_rng_state(),ts)
    assert capture(Client(policy),inputs(tmp_path/'in',2),tmp_path/'out',batch_size=2,comparison='serial_vs_batch')=='mismatch'
    assert not reaudit_case(tmp_path/'out/raw-case-0000')['passed']
    assert validate_run(tmp_path/'out',expected_batch_size=2,require_passed=False)['status']=='mismatch'
    with pytest.raises(ValueError,match='Same-shape'):validate_run(tmp_path/'out',expected_batch_size=2)


@pytest.mark.parametrize('bad',['packing','splitting'])
def test_same_shape_reference_detects_shared_batch_corruption(monkeypatch,bad):
    import rosa_repro.batching as shared
    if bad=='packing':
        original=shared.combine_observations
        monkeypatch.setattr(shared,'combine_observations',lambda obs:original(list(reversed(obs))))
    else:
        original=shared.split_actions
        monkeypatch.setattr(shared,'split_actions',lambda actions,n:list(reversed(original(actions,n))))
    r=collect_arrays(DummyPolicy(),[observation(1),observation(9)],seed=42)
    assert not r['passed'] and r['rng_alignment']=='identical_seed_and_shape'


def test_even_matching_cross_shape_diagnostic_cannot_certify_adapter(tmp_path):
    capture(Client(),inputs(tmp_path/'in',4),tmp_path/'out',batch_size=4,comparison='serial_vs_batch')
    r=validate_run(tmp_path/'out',expected_batch_size=4,require_passed=False)
    assert r['status']=='passed' and not r['numerical_admission_eligible']
    with pytest.raises(ValueError,match='Same-shape'):validate_run(tmp_path/'out',expected_batch_size=4)


def test_default_rng_restored_after_reference_or_candidate_failure(monkeypatch):
    import rosa_repro.action_regression as module
    def fail(*args):
        random.random();np.random.random();torch.rand(1);raise RuntimeError('injected model failure')
    for name in ('official_batch_reference','predict_policy_batch'):
        py=random.getstate();ns=np.random.get_state();ts=torch.random.get_rng_state().clone()
        with monkeypatch.context() as patch:
            patch.setattr(module,name,fail)
            with pytest.raises(RuntimeError):collect_arrays(DummyPolicy(),[observation(1),observation(2)],seed=42)
        assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1]) and torch.equal(torch.random.get_rng_state(),ts)


@pytest.mark.parametrize('kind',['array','summary','missing','extra','controls','source'])
def test_modified_or_incomplete_numerical_evidence_rejected(tmp_path,kind):
    out=tmp_path/'out';capture(Client(),inputs(tmp_path/'in',2),out)
    if kind=='array':np.savez_compressed(out/'raw-case-0000/candidate-000.npz',**{'action.x':np.zeros((1,8,1),np.float32)})
    elif kind=='summary':
        p=out/'comparison_summary.json';r=json.loads(p.read_text());r['results'][0]['passed']=False;p.write_text(json.dumps(r));reseal(out)
    elif kind=='missing':(out/'case-0001.json').unlink();reseal(out)
    elif kind=='extra':(out/'extra.txt').write_text('unexpected')
    elif kind=='controls':
        p=out/'run_context.json';r=json.loads(p.read_text());r['seed']+=1;p.write_text(json.dumps(r));reseal(out)
    else:
        p=out/'server_evidence.json';r=json.loads(p.read_text());r['rosa_source_manifest']={};p.write_text(json.dumps(r));reseal(out)
    with pytest.raises(ValueError):validate_run(out)


def test_interrupted_capture_retains_failure_and_completed_cases(tmp_path):
    class Broken(Client):
        def call_endpoint(self,name,params=None,**kwargs):
            if params and params['seed']>42:raise TimeoutError('fixture transport timeout')
            return super().call_endpoint(name,params,**kwargs)
    out=tmp_path/'out';assert capture(Broken(),inputs(tmp_path/'in',2),out)=='failed'
    summary=json.loads((out/'comparison_summary.json').read_text())
    assert summary['cases']==1 and summary['expected_cases']==2 and summary['failed_or_timed_out_request_execution_unknown']
    assert reaudit_case(out/'raw-case-0000')['passed']
    assert validate_run(out,require_passed=False)['status']=='failed'
    with pytest.raises(ValueError):validate_run(out)


def test_no_discarded_tail_no_output_overwrite_or_changed_tolerances(tmp_path):
    source=inputs(tmp_path/'in',3)
    with pytest.raises(ValueError,match='divisible'):capture(Client(),source,tmp_path/'bad',batch_size=2)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError):capture(Client(),source,tmp_path/'bad',atol=-1)
    class Bad(Client):
        def call_endpoint(self,name,params=None,**kwargs):
            r=super().call_endpoint(name,params,**kwargs)
            if params:r['atol']=.1
            return r
    assert capture(Bad(),source,tmp_path/'out')=='failed'
    with pytest.raises(FileExistsError):capture(Client(),source,tmp_path/'out')
