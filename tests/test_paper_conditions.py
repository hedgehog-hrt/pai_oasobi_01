from copy import deepcopy
import pytest
from rosa_repro.paper_conditions import allocation,generate,validate,workload

def test_r05_grid_no_fake_deadline_and_independent_confirmation():
 d=generate();r=[c for c in d['cells'] if c['experiment']=='R05']
 assert len(r)==99 and len({c['group'] for c in r})==9
 assert all(c['latency_slo_ms'] is None and c['frequency_cap_hz'] is None for c in r)
 assert sum(c['stage']=='pilot' for c in r)==45
 assert sum(c['batch_size'] is None for c in r)==27
 assert validate(d)==d and not d['executable']
 assert set(d['randomized_order'])=={c['id'] for c in d['cells']}

def test_table3_and_generic_rounding_and_shortage():
 assert list(allocation(('s1','s2','safety','monitor'),8,'weighted')['counts'].values())==[1,3,1,3]
 assert allocation(('s1','s2','safety','monitor'),8,'equal')['counts']==dict(s1=2,s2=2,safety=2,monitor=2)
 assert allocation(('s1','monitor'),3,'equal')['counts']==dict(s1=1,monitor=2)
 assert allocation(('s1','monitor'),1,'equal')['status']=='infeasible'
 assert allocation(('s1','monitor'),3,'weighted')['status']=='unresolved'
 assert allocation(('s1','monitor'),6,'weighted',dict(s1=1,monitor=3))['counts']==dict(s1=2,monitor=4)

def test_population_and_separate_ablation():
 d=generate();assert sum(c['experiment']=='R09' and c['comparison']=='ROSA' for c in d['cells'])==24
 assert [c['robots'] for c in workload('Mix2',8)]==[4,4]
 assert [c['pipeline'] for c in workload('Mix4',16)]==['P1','P2','P3','P4']
 for c in d['cells']:
  if c.get('comparison')=='placement_only':
   assert c['s1_batch_size'] is None and c['fixed_ROSA_batch_and_rates_sha256'] is None
   assert c['s1_s2_rate_control']=='identical_to_bound_ROSA_reference'
  if c.get('comparison')=='full_baseline':assert c['s1_batch_size']==1

def test_one_gpu_alias_and_infeasible_preserved():
 d=generate(gpu_count=1);r=[c for c in d['cells'] if c.get('workload')=='P1']
 assert len(r)==36 and all('weighted' in c['aliases'] for c in r if c['method']=='equal')
 assert any(c.get('allocation',{}).get('status')=='infeasible' for c in d['cells'] if c.get('allocation'))

def test_selection_binding_and_same_best_alias():
 d=generate(r05_selection={'k1-a50':dict(batch_size=1,prediction_evidence_sha256='a'*64)})
 assert len([c for c in d['cells'] if c.get('group')=='k1-a50'])==8
 assert sum('predicted_best' in c['aliases'] for c in d['cells'])==3
 with pytest.raises(ValueError):generate(r05_selection={'k1-a50':dict(batch_size=3,prediction_evidence_sha256='a'*64)})

@pytest.mark.parametrize('change',['deadline','omission','order','controls'])
def test_mutation_rejected(change):
 d=deepcopy(generate())
 if change=='deadline':d['cells'][0]['latency_slo_ms']=200
 if change=='omission':d['cells'].pop()
 if change=='order':d['randomized_order'][0]=d['randomized_order'][1]
 if change=='controls':d['cells'][-1]['profile_ids']=['invented']
 with pytest.raises(ValueError):validate(d)

@pytest.mark.parametrize('args',[dict(repetitions=2),dict(seed=True),dict(gpu_count=0),dict(sizes={'s1':0,'monitor':1,'safety':1,'s2':1})])
def test_invalid_design(args):
 with pytest.raises(ValueError):generate(**args)
