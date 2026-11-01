import json
import numpy as np
import pytest
from rosa_repro.adapter import ProtocolError
from rosa_repro.dummy import DummyEpisode,DummyTransport
from rosa_repro.paired import save_state_bank,load_state_bank,run_paired
from rosa_repro.contracts import validate,MEASUREMENT_SCHEMA


def bank(path,n=2):
    save_state_bank(path,[np.array([i+1.,.3]) for i in range(n)],task='CPU fixture',source={'kind':'synthetic'})
    return path


def test_paired_conditions_outcomes_and_actual_actions(tmp_path):
    b=bank(tmp_path/'bank'); out=tmp_path/'out'
    result=run_paired(bank=b,output=out,env_factory=DummyEpisode,transport_factory=DummyTransport,execution_kind='cpu_dummy')
    assert result['paired_comparison_valid'] and result['started_episodes']==4
    assert all(v['successful']==2 for v in result['arms'].values())
    records=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    for r in records: validate(r,MEASUREMENT_SCHEMA)
    assert sorted(r['executed_steps'] for r in records if r['record_type']=='action')==[3]*4+[8]*4
    assert len(list(out.glob('*/*/observation-*/observation.json')))==8
    assert len(list(out.glob('*/*/action-*.npz')))==8


def test_failures_remain_in_all_started_denominator(tmp_path):
    def transport(arm):
        t=DummyTransport(arm)
        if arm=='shared':
            def fail(*args): raise TimeoutError('fixture')
            t.predict=fail
        return t
    result=run_paired(bank=bank(tmp_path/'bank'),output=tmp_path/'out',env_factory=DummyEpisode,
        transport_factory=transport,execution_kind='cpu_dummy')
    assert result['arms']['shared']==dict(started=2,completed=0,successful=0,errors=2)
    assert not result['paired_comparison_valid']


def test_comparison_refuses_mismatched_initial_observations(tmp_path):
    count=0
    def factory():
        nonlocal count
        e=DummyEpisode(); count+=1
        if count==2:
            reset=e.reset
            def changed(state,seed): return reset(state+1,seed)
            e.reset=changed
        return e
    r=run_paired(bank=bank(tmp_path/'bank',1),output=tmp_path/'out',env_factory=factory,
        transport_factory=DummyTransport,execution_kind='cpu_dummy')
    assert not r['paired_comparison_valid'] and not r['pairs'][0]['initial_conditions_match']


def test_step_budget_exact_and_no_overwrite(tmp_path):
    b=bank(tmp_path/'bank',1); out=tmp_path/'out'
    r=run_paired(bank=b,output=out,env_factory=DummyEpisode,transport_factory=DummyTransport,
        execution_kind='cpu_dummy',max_steps=9)
    outcome=json.loads((out/'T01-0000/shared/outcome.json').read_text())
    assert outcome['executed_steps']==9 and outcome['termination_reason']=='step_limit' and not outcome['success']
    with pytest.raises(FileExistsError):
        run_paired(bank=b,output=out,env_factory=DummyEpisode,transport_factory=DummyTransport,execution_kind='cpu_dummy')


def test_corrupt_state_bank_refused_before_environment_created(tmp_path):
    b=bank(tmp_path/'bank')
    with (b/'states.npz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):load_state_bank(b)


def test_duplicate_cases_refused(tmp_path):
    b=bank(tmp_path/'bank')
    m=json.loads((b/'initial_states.json').read_text());m['cases'][1]['case_id']=m['cases'][0]['case_id']
    (b/'initial_states.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='case ID'):load_state_bank(b)


def test_capture_initial_observations_without_inference(tmp_path):
    from rosa_repro.paired import capture_initial_observations
    capture_initial_observations(bank(tmp_path/'bank'),tmp_path/'observations',DummyEpisode)
    summary=json.loads((tmp_path/'observations/capture_summary.json').read_text())
    assert summary['cases']==2 and summary['policy_calls']==0 and summary['executed_steps']==0
