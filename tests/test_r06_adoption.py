import asyncio,json,sys
from copy import deepcopy
import pytest
from rosa_repro.r06_adoption import quality,assess,review_checks,validate_review
from rosa_repro.r06_protocol import prepare,TASKS
from rosa_repro.r06_runtime import run_cell
from rosa_repro.statistical_evaluation import r06_report
from rosa_repro.profile_adoption import Evidence
from rosa_repro.regression import write_json
from test_r06 import banks,Sim


def test_required_r06_cannot_be_omitted(tmp_path):
 from test_profile_adoption import s1_quality_fixture
 from rosa_repro.profile_adoption import _quality
 q,c,s=s1_quality_fixture(tmp_path,1)
 with pytest.raises(KeyError,match='r06'):_quality(Evidence(tmp_path),q,c,'fixture',s)


@pytest.mark.parametrize('batch',[8,16])
def test_large_candidate_adds_real_full_batch_condition(tmp_path,batch):
 root=banks(tmp_path,n=batch+1);s=prepare(root,mode='cpu_dummy',phase='formal',selected_batch=batch)
 assert {c['concurrency'] for c in s['cells']}=={1,2,4,batch} and len(s['cells'])==36
 r=asyncio.run(run_cell(s,root,f'T01-{batch}-selected',tmp_path/'raw',sim_python=sys.executable,sim_factory=Sim))
 assert r['batch_sizes']==[1,batch]


def cpu_report(tmp_path):
 root=banks(tmp_path,n=5);s=prepare(root,mode='cpu_dummy',phase='formal',selected_batch=4);mapping={}
 for cell in s['cells']:
  mapping[cell['id']]=cell['id']
  asyncio.run(run_cell(s,root,cell['id'],tmp_path/cell['id'],sim_python=sys.executable,sim_factory=Sim))
 return s,root,mapping,r06_report(s,tmp_path,mapping)


def test_raw_report_uncertainty_coverage_and_explicit_review(tmp_path):
 s,root,mapping,report=cpu_report(tmp_path)
 assert report['status']=='complete' and len(report['comparisons'])==18
 write_json(tmp_path/'suite.json',s);write_json(tmp_path/'map.json',mapping)
 q=dict(r06=dict(suite='suite.json',bank_root='banks',directories='map.json'))
 with pytest.raises(ValueError,match='formal GPU'):quality(Evidence(tmp_path),q,{'batch_size':4},None)
 with pytest.raises(ValueError,match='formal GPU'):assess(report,4)
 # Metadata fabrication stays inside this contract test; CPU evidence is never adopted.
 mock_gpu=deepcopy(report);mock_gpu['mode']='libero_gpu'
 a=assess(mock_gpu,4);assert len(a['unresolved_comparisons'])==18 and not a['automatic_quality_equivalence_claim']
 checks=review_checks(a)
 with pytest.raises(ValueError,match='incomplete'):validate_review(a,checks)
 for c in checks.values():c.update(accepted=True,reason='TEST ONLY acknowledges unresolved interval')
 validate_review(a,checks)
 changed=deepcopy(checks);next(iter(changed.values()))['confidence_interval']=[0,0]
 with pytest.raises(ValueError,match='statistics changed'):validate_review(a,changed)
 for row in mock_gpu['rows']:
  if row['cell']['task']=='T01':row['audit']['batch_sizes']=[1]
 with pytest.raises(ValueError,match='never observed'):assess(mock_gpu,4)
 (tmp_path/mapping['T01-1-official']/'result.json').write_text('{}')
 incomplete=r06_report(s,tmp_path,mapping);incomplete['mode']='libero_gpu'
 with pytest.raises(ValueError,match='complete formal'):assess(incomplete,4)


def test_final_adoption_requires_r06_specific_review(monkeypatch,tmp_path):
 from rosa_repro import profile_adoption as gate
 from rosa_repro.statistical_evaluation import paired_success
 states=[dict(task='T01',state_sha256=str(i),seed=i,status='completed',success=True,initial_observation_sha256=str(i),restored_state_sha256=str(i)) for i in range(50)]
 stats=paired_success(states,states)
 quality_value={'report':{'comparisons':[dict(candidate='T01-4-selected',statistics=stats)]}}
 inspection=dict(evidence_set_sha256='bound',config={'role':'s1'},quality={'r06':quality_value})
 monkeypatch.setattr(gate,'inspect',lambda *args:deepcopy(inspection))
 r=gate.review_template(inspection);r['reviewer']='TEST ONLY'
 for v in r['checks'].values():v.update(accepted=True,reason='TEST ONLY')
 with pytest.raises(ValueError,match='R06 comparison review'):gate.adopt({},tmp_path,r)
 r['r06_checks']['T01-4-selected'].update(accepted=True,reason='TEST ONLY explicitly accepts unresolved interval, not equivalence')
 assert gate.adopt({},tmp_path,r)['status']=='accepted'


def test_observed_server_change_cannot_be_resealed_as_complete(tmp_path):
 from test_r06 import run
 from rosa_repro.r06_runtime import seal,reaudit
 r,out,s=run(tmp_path)
 (out/'server_after.json').write_text('{}');(out/'evidence.json').unlink();seal(out)
 with pytest.raises(ValueError,match='Observed R06 server'):reaudit(out)
