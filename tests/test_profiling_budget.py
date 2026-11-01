import asyncio
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from test_profile_preparation import config,corpus
from rosa_repro.profile_protocol import make_suite,validate_suite,template
from rosa_repro.profiling_budget import formal_timing,default_timing_policy,validate_formal_case


def test_half_hertz_gets_1000_samples_without_lowering_acceptance_threshold():
    c=template('s2');t=formal_timing(.5,c)
    assert t['executable'] and t['protocol']['measure_min_success']==1000
    assert t['protocol']['measure_min_s']==120 and t['protocol']['measure_max_s']==2446
    assert t['phases']['measure']['ideal_arrival_time_s']==2000
    assert t['bounded_phase_and_drain_s']==2931


@pytest.mark.parametrize('rate',[.01,.001,1e-6,10000])
def test_infeasible_time_or_request_budgets_are_explicit(rate):
    t=formal_timing(rate,template('s2'));assert not t['executable'] and t['blocking_reasons']


def test_explicit_policy_can_extend_but_never_mutates_during_execution():
    c=template('s2');p=default_timing_policy();p.update(warmup_hard_cap_s=86400,measure_hard_cap_s=172800)
    t=formal_timing(.01,c,p);assert t['executable'] and t['protocol']['measure_max_s']==120046
    case=dict(offered_hz=.01,protocol=t['protocol'],timing=t);validate_formal_case(case,c)
    case=deepcopy(case);case['protocol']['measure_min_success']=1
    with pytest.raises(ValueError):validate_formal_case(case,c)


@pytest.mark.parametrize('rate',[0,-1,float('inf'),float('nan'),True,1e-200])
def test_bad_rates_do_not_create_nonfinite_or_unbounded_reports(rate):
    with pytest.raises(ValueError):formal_timing(rate,template('s2'))


def test_suite_accounts_for_every_rate_repetition_and_budget(tmp_path):
    _,book=corpus(tmp_path);c=config()
    s=make_suite(c,book,stage='profile',phase='formal',rates=[.5,1,10],repetitions=3)
    validate_suite(s);assert s['executable'] and len(s['cases'])==9
    assert s['estimated_max_trial_s']==sum(x['timing']['bounded_phase_and_drain_s'] for x in s['cases'])
    blocked=make_suite(c,book,stage='profile',phase='formal',rates=[.5,1,10],repetitions=3,budget_limit_s=1)
    validate_suite(blocked);assert not blocked['executable']
    assert blocked['unresolved'][-1]['reasons']==['suite_time_budget_exceeded']


def test_unresolved_budget_stops_before_corpus_or_ray(tmp_path):
    _,book=corpus(tmp_path)
    s=make_suite(config(),book,stage='profile',phase='formal',rates=[.001])
    path=Path(__file__).resolve().parents[1]/'scripts/postpod_models.py'
    spec=importlib.util.spec_from_file_location('postpod_budget_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(ValueError,match='unresolved timing'):
        asyncio.run(m.execute(s,tmp_path/'never-created',tmp_path/'missing-corpus'))
    assert not (tmp_path/'never-created').exists()
