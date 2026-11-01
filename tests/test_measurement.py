import copy

import pytest
from jsonschema import ValidationError

from rosa_repro.measurement import evaluate_p1
from helpers import request, action


def evaluate(records, **kw):
    context = dict(run_id='run',clock_domain='host-boot-1',window_start_ns=0,window_end_ns=2_000_000_000)
    context.update(kw)
    return evaluate_p1(records,**context)


def test_counts_chunks_not_control_steps_and_failures_not_hidden():
    good=request('good'); late=request('late',sent=500_000_000,duration_ms=201)
    error=request('failed',sent=1_000_000_000,duration_ms=1000,status='timeout',valid=False)
    result=evaluate([good,late,error,action(good),action(late)])
    assert result['requests']==3
    assert result['raw_action_chunks']==2 and result['qualified_action_chunks']==1
    assert result['qualified_action_rate_hz']==.5
    assert result['deadline_unqualified_requests']==2
    assert result['latency_ms_successful_valid_only']['p99']==201
    assert result['qualified_action_chunks']>0 # Even when p99 exceeds 200ms.


def test_exact_deadline_and_window_boundary():
    r=request(duration_ms=200)
    result=evaluate([r,action(r)])
    assert result['deadline_qualified_requests']==1
    a=action(r,start=2_000_000_000)
    assert evaluate([r,a])['qualified_action_chunks']==0


def test_warmup_and_left_boundary_are_excluded():
    r=request(warmup=True)
    assert evaluate([r,action(r)])['requests']==0
    r=request()
    result=evaluate([r,action(r,start=600_000_000)],window_start_ns=500_000_000)
    assert result['qualified_action_chunks']==0
    assert result['cross_boundary_or_warmup_actions_excluded']==1


def test_right_censored_requests_are_not_dropped():
    r=request(sent=1_900_000_000,duration_ms=500,status='timeout',valid=False)
    result=evaluate([r])
    assert result['requests']==1 and result['deadline_unqualified_requests']==1


@pytest.mark.parametrize('case',['duplicate_request','duplicate_action','episode','clock','component','deadline','timestamps','nan','extra','invalid_apply'])
def test_reject_invalid_measurements(case):
    r=request(); a=action(r); records=[r,a]
    if case=='duplicate_request': records.append(copy.deepcopy(r))
    if case=='duplicate_action': records.append(copy.deepcopy(a))
    if case=='episode': a['episode_id']='reset-episode'
    if case=='clock': r['clock_domain']='other-host'
    if case=='component': r['component_id']='monitor'
    if case=='deadline': r['deadline_ms']=500
    if case=='timestamps': a['start_ns']=0
    if case=='nan': r['deadline_ms']=float('nan')
    if case=='extra': r['unknown_field']=True
    if case=='invalid_apply': r['valid_output']=False
    with pytest.raises((ValueError,ValidationError)):
        evaluate(records)


def test_no_action_is_not_inferred_from_request():
    result=evaluate([request()])
    assert result['deadline_qualified_requests']==1
    assert result['qualified_action_chunks']==0
