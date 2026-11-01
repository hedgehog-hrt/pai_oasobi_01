import asyncio,json,math
from copy import deepcopy
import numpy as np
import pytest
from rosa_repro.r05_load_curves import solve,prepare,curve_plan,validate_plan,generate
from rosa_repro.r05_prediction import verify
from rosa_repro.profile_protocol import load_corpus
from rosa_repro.profile_runner import run_probe,profile_candidate
from rosa_repro.r05_workflow import check_pilots
from rosa_repro.r05_protocol import prepare as confirm,timing
from rosa_repro.paper_conditions import generate as paper
from rosa_repro.execution_plan import digest
from test_r05 import confirmation,Transport


def test_exact_curve_fixed_point_and_right_censoring():
 p=[dict(rate_hz=1,mean_ms=100),dict(rate_hz=20,mean_ms=100)]
 r=solve(p,1,.1);assert r['predicted_total_action_rate_hz']==pytest.approx(5) and not r['right_censored']
 assert solve(p,16,.1)['right_censored']
 assert solve([dict(rate_hz=10,mean_ms=1000),dict(rate_hz=20,mean_ms=1000)],1,1)['status']=='unresolved'
 with pytest.raises(ValueError):solve(p[:1],1,.1)


def test_nonmonotonic_curve_all_segments_checked_against_dense_oracle():
 p=[dict(rate_hz=1,mean_ms=10),dict(rate_hz=5,mean_ms=1000),dict(rate_hz=15,mean_ms=10),dict(rate_hz=30,mean_ms=300)]
 r=solve(p,2,.05);grid=np.linspace(1,30,100000);lat=np.interp(grid,[x['rate_hz'] for x in p],[x['mean_ms'] for x in p])/1000
 feasible=grid[grid*(.05+lat)<=2]
 assert r['predicted_total_action_rate_hz']==pytest.approx(feasible[-1],abs=.0004) and r['predicted_total_action_rate_hz']>15


def evidence(tmp_path):
 s,root,pilots=confirmation(tmp_path);pilot=json.loads((tmp_path/pilots['k4-a50'][0]/'suite.json').read_text());collection=prepare(pilot,[10.,20.],1)
 _,samples=load_corpus(pilot['corpus'],root,'profile','cpu_dummy');paths={'k4-a50':{str(b):[] for b in (1,2,4,8,16)}}
 async def go():
  for case in collection['cases']:
   plan=curve_plan(pilot['config'],case['r05_cell']);validate_plan(plan)
   assert plan['config']['streams']==4
   out=tmp_path/case['case_id'];t=Transport(plan)
   r=await run_probe(plan,t,samples,case,out,corpus_id=pilot['corpus']['corpus_id'],split='profile')
   if t.task:await t.task
   assert r['status']=='complete'
   paths['k4-a50'][str(case['batch_cap'])].append(out.name)
 asyncio.run(go());return s,root,pilots,paths


def test_real_core_profiles_prediction_and_confirmation_binding(tmp_path):
 s,root,pilots,paths=evidence(tmp_path);p=generate(tmp_path,pilots,paths)
 assert p['method']=='measured_load_curve_fixed_point_v1' and verify(p,tmp_path)==p
 assert all(r['right_censored'] for r in p['details']['k4-a50']['rows'])
 text=json.dumps(p);import hashlib
 h=hashlib.sha256(text.encode()).hexdigest();chosen={'k4-a50':dict(batch_size=p['groups']['k4-a50'],prediction_evidence_sha256=h)}
 new=confirm(paper(r05_selection=chosen),s['config'],s['corpus'],stage='confirmation',groups=['k4-a50'],protocol=s['protocol'],prediction_artifacts={h:text})
 assert check_pilots(new,tmp_path,pilots)
 bad=deepcopy(p);bad['groups']['k4-a50']=16
 with pytest.raises(ValueError):verify(bad,tmp_path)
 first=paths['k4-a50']['1'][0];assert 'R05_calibration_not_R07_adoption' in profile_candidate([tmp_path/first])['blocking_reasons']
 wrong=deepcopy(paths);wrong['k4-a50']['16']=wrong['k4-a50']['1']
 with pytest.raises(ValueError):generate(tmp_path,pilots,wrong)
