from copy import deepcopy
import json,math
import pytest
from rosa_repro.statistical_evaluation import exact_binomial_interval,paired_success,latency_blocks,r06_report,latency_report
from test_r06 import run

def outcomes(values):
 return [dict(task='T01',state_sha256=str(i),seed=i,status='completed' if v is not None else 'error',success=v,
  initial_observation_sha256=str(i),restored_state_sha256=str(i)) for i,v in enumerate(values)]

def test_clopper_pearson_independent_reference_and_boundaries():
 # Published SciPy exact-binomial example (k=7, n=50), independent of this implementation.
 assert exact_binomial_interval(7,50)==pytest.approx([.05819170033997342,.26739600249700846],abs=1e-12)
 assert exact_binomial_interval(0,50)==pytest.approx([0,1-.025**(1/50)])
 assert exact_binomial_interval(50,50)==pytest.approx([.025**(1/50),1])
 assert exact_binomial_interval(1,1)==pytest.approx([.025,1])

def test_all_fifty_successes_does_not_establish_five_point_equivalence():
 a=outcomes([True]*50);r=paired_success(a,a)
 assert r['success_difference']==0 and r['decision']=='unresolved'
 assert r['confidence_interval'][0]<-.05 and r['confidence_interval'][1]>.05
 assert r['confidence_interval']==pytest.approx([-(1-.0125**(1/50)),1-.0125**(1/50)])

def test_more_cases_can_bound_discordance_under_stated_assumption():
 a=outcomes([True]*200);r=paired_success(a,a)
 assert r['decision']=='within_margin_under_sampling_assumption'
 assert 0<r['confidence_interval'][1]<.05

def test_paired_discordance_not_independent_proportions_and_order_invariant():
 a=outcomes([True,True,False,False]*5);b=outcomes([False,True,True,False]*5)
 r=paired_success(a,list(reversed(b)));assert r['discordant_gain']==5 and r['discordant_loss']==5
 assert r['success_difference']==0

def test_drop_rule_and_missing_not_dropped():
 a=outcomes([True]*50);b=outcomes([False]*3+[True]*47);assert paired_success(a,b)['decision']=='investigate_drop'
 b=outcomes([None]+[True]*49);r=paired_success(a,b)
 assert not r['all_pairs_valid'] and r['confidence_interval'] is None and r['planned_pairs']==50
 assert r['candidate_successful_all_planned']==49 and r['missing_outcome_identification_bounds']==[-.02,0]
 assert paired_success(a,a,trial_complete=False)['confidence_interval'] is None

def test_mismatched_initial_conditions_and_duplicate_case_rejected():
 a=outcomes([True,False]);b=deepcopy(a);b[0]['initial_observation_sha256']='wrong'
 r=paired_success(a,b);assert r['initial_condition_mismatches'] and r['confidence_interval'] is None
 with pytest.raises(ValueError):paired_success(a,[a[0],a[0]])
 with pytest.raises(ValueError):paired_success(a,[dict(a[0],seed=42),a[1]])

def rows(n=16):
 return [dict(request_id=f'r{i}-{j}',cohort_ns=i*100+10+j,status='ok',latency_ms=float(i+1)) for i in range(n) for j in range(5)]

def test_seeded_time_block_resampling_and_quantiles():
 a=latency_blocks(rows(),start_ns=0,end_ns=1600,block_ns=100,resamples=200,seed=42)
 b=latency_blocks(list(reversed(rows())),start_ns=0,end_ns=1600,block_ns=100,resamples=200,seed=42)
 assert a==b and a['status']=='estimated' and a['total_blocks']==16 and a['successful']==80
 assert a['metrics']['p50']['estimate_ms']==8 and a['metrics']['p99']['estimate_ms']==16
 assert a['metrics']['p50']['confidence_interval_ms'][0]<8<a['metrics']['p50']['confidence_interval_ms'][1]

def test_block_resampling_keeps_synchronous_replicas_together():
 one=[r for r in rows() if r['request_id'].endswith('-0')]
 a=latency_blocks(one,start_ns=0,end_ns=1600,block_ns=100,resamples=200)
 b=latency_blocks(rows(),start_ns=0,end_ns=1600,block_ns=100,resamples=200)
 assert a['metrics']==b['metrics'] # Repeated robots do not manufacture extra precision.

def test_failed_latency_omitted_but_denominator_and_empty_blocks_retained():
 x=rows();x.append(dict(request_id='missing',cohort_ns=1,status='not_sent',latency_ms=None))
 r=latency_blocks(x,start_ns=0,end_ns=2000,block_ns=100,resamples=200)
 assert r['planned_logged_arrivals']==81 and r['successful']==80 and r['status_counts']['not_sent']==1
 assert r['total_blocks']==20 and r['block_arrival_counts'][-1]==0

def test_insufficient_blocks_and_no_independent_fallback():
 r=latency_blocks(rows(4),start_ns=0,end_ns=400,block_ns=100,resamples=100)
 assert r['status']=='unavailable' and all(x['confidence_interval_ms'] is None for x in r['metrics'].values())

@pytest.mark.parametrize('bad',['duplicate','invalid','outside','failure_latency','few_resamples'])
def test_invalid_latency_analysis(bad):
 x=rows();kwargs=dict(start_ns=0,end_ns=1600,block_ns=100,resamples=100)
 if bad=='duplicate':x.append(x[0])
 if bad=='invalid':x[0]['latency_ms']=float('nan')
 if bad=='outside':x[0]['cohort_ns']=-1
 if bad=='failure_latency':x[0]['status']='timeout'
 if bad=='few_resamples':kwargs['resamples']=99
 with pytest.raises(ValueError):latency_blocks(x,**kwargs)

def test_r06_report_retains_expected_missing_cells(tmp_path):
 r,out,s=run(tmp_path,cell='T01-4-selected')
 directories={c['id']:'missing-'+c['id'] for c in s['cells']};directories['T01-4-selected']='out'
 report=r06_report(s,tmp_path,directories)
 assert len(report['rows'])==27 and sum(x['status']=='missing' for x in report['rows'])==26
 assert len(report['comparisons'])==18 and report['status']=='incomplete'
 assert all(r['statistics'] is None for r in report['comparisons'])

def test_pipeline_adapter_reaudits_raw_and_retains_failure(tmp_path):
 from test_pipeline_runtime import run as pipeline_run,bindings
 from runtime_helpers import plan_for
 pipeline_run(plan_for(bindings=bindings(fail_calls=[4])),tmp_path/'pipeline')
 a=json.loads((tmp_path/'pipeline/audit.json').read_text());block=a['window_end_ns']-a['window_start_ns']
 r=latency_report(tmp_path/'pipeline',kind='pipeline',block_ns=block,resamples=100)
 assert r['trial_status']=='failed' and r['status']=='insufficient' and r['periodic_slot_metrics'] is not None
 assert any(g['status_counts'].get('error',0)>0 for g in r['groups'].values())

def test_profile_adapter_uses_scheduled_arrivals_and_sealed_evidence(tmp_path):
 # Reuse the real CPU probe fixture, never construct trusted profile summaries by hand.
 from test_profile_preparation import run as profile_run
 r,out=profile_run(tmp_path)
 window=r['measurement'];block=window['end_ns']-window['start_ns']
 result=latency_report(out,kind='profile',block_ns=block,resamples=100)
 g=result['groups']['component']
 assert g['planned_logged_arrivals']==r['metrics']['arrivals'] and g['successful']==r['metrics']['successful']
 assert 'not_sent' in result['cohort_definition'] and result['status']=='insufficient'
 with (out/'requests.jsonl').open('a') as f:f.write('{}\n')
 with pytest.raises(ValueError):latency_report(out,kind='profile',block_ns=block,resamples=100)


def test_partial_final_target_block_equalized_without_dropping_tail():
 x=rows();x.append(dict(request_id='tail',cohort_ns=1649,status='timeout',latency_ms=None))
 r=latency_blocks(x,start_ns=0,end_ns=1650,block_ns=100,resamples=100)
 assert r['total_blocks']==16 and r['actual_block_duration_ns_range']==[103,104]
 assert r['planned_logged_arrivals']==81 and sum(r['block_arrival_counts'])==81
 assert r['block_arrival_counts'][-1]==1


def test_sparse_empty_resample_suppresses_interval_instead_of_dropping_it():
 x=[dict(request_id=str(i),cohort_ns=i*100,status='ok',latency_ms=1.) for i in range(8)]
 r=latency_blocks(x,start_ns=0,end_ns=100000,block_ns=100,resamples=10000,seed=42)
 assert r['empty_resamples']>0 and r['status']=='unavailable'
 assert r['reason']=='resamples_without_success_no_conditional_dropping'
