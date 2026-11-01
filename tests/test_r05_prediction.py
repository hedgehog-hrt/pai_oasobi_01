import json
from copy import deepcopy
import pytest
from rosa_repro.r05_prediction import rank,generate,verify
from rosa_repro.r05_workflow import check_pilots
from test_r05 import confirmation


def test_equation_four_tradeoff_measured_caps_and_tie_rule():
 p=[dict(batch_cap=b,mean_latency_ms=m) for b,m in zip((1,2,4,8,16),(400,250,100,100.5,101))]
 r=rank(p,robots=4,action_time_s=.05)
 assert r['selected_batch_cap']==4 and r['maximum_point_estimate_hz']==pytest.approx(4/.15)
 assert r['tie_eligible_batch_caps']==[4,8,16]
 p[0]['mean_latency_ms']=99.8
 assert rank(p,robots=1,action_time_s=1)['selected_batch_cap']==1
 with pytest.raises(ValueError):rank(p[:-1],robots=4,action_time_s=.05)
 p[0]['mean_latency_ms']=float('nan')
 with pytest.raises(ValueError):rank(p,robots=4,action_time_s=.05)


def test_actual_pilot_prediction_recomputed_and_tampering_refused(tmp_path):
 s,root,paths=confirmation(tmp_path);pred=json.loads(next(iter(s['prediction_artifacts'].values())))
 assert verify(pred,tmp_path)==pred and pred==generate(tmp_path,paths)
 assert pred['latency_slo_ms'] is None and not pred['optimality_claim']
 assert all(set(r['actual_batch_sizes'])<={1,2,3,4} for r in pred['details']['k4-a50']['rows'])
 altered=deepcopy(pred);altered['details']['k4-a50']['rows'][0]['mean_latency_ms']=0
 with pytest.raises(ValueError,match='recomputed'):verify(altered,tmp_path)
 altered=deepcopy(pred);altered['source_evidence_sha256'][0]='0'*64
 with pytest.raises(ValueError,match='recomputed'):verify(altered,tmp_path)
 with pytest.raises(ValueError):generate(tmp_path,{'k4-a50':paths['k4-a50'][:-1]})
 with pytest.raises(ValueError):generate(tmp_path,{'k4-a50':[paths['k4-a50'][0]]*5})
 with pytest.raises(ValueError,match='matching pilot'):generate(tmp_path,{'k1-a50':paths['k4-a50']})
 (tmp_path/paths['k4-a50'][0]/'events.jsonl').write_text('{}\n')
 with pytest.raises(ValueError):check_pilots(s,tmp_path,paths)


def test_confirmation_cannot_be_used_for_prediction(tmp_path,monkeypatch):
 s,root,paths=confirmation(tmp_path)
 from rosa_repro import r05_prediction
 original=r05_prediction.validate_suite
 def confirmation_instead(prior):
  v=original(prior);v['stage']='confirmation';return v
 monkeypatch.setattr(r05_prediction,'validate_suite',confirmation_instead)
 with pytest.raises(ValueError,match='confirmation cannot'):generate(tmp_path,paths)


def test_pilot_runtime_mutation_even_after_resealing_is_rejected(tmp_path):
 from rosa_repro.r05_runtime import seal
 s,root,paths=confirmation(tmp_path);p=tmp_path/paths['k4-a50'][0]
 r=json.loads((p/'result.json').read_text());r['worker_after']['r05-s1']['backend_attestation']['runtime_fingerprint']='changed'
 (p/'result.json').write_text(json.dumps(r));(p/'evidence.json').unlink();seal(p)
 with pytest.raises(ValueError):generate(tmp_path,paths)
