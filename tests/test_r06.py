import asyncio,json,sys
import numpy as np
import pytest
from rosa_repro.r06_protocol import TASKS,prepare,validate
from rosa_repro.r06_service import RegressionBatchWorker
from rosa_repro.r06_runtime import run_cell,reaudit,CPUTransport,seal
from rosa_repro.paired import save_state_bank
from rosa_repro.dummy import DummyPolicy,observation
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.simulator_process import ChangingDummyEpisode


def banks(tmp_path,n=5):
 root=tmp_path/'banks'
 for task,name in TASKS.items():
  save_state_bank(root/task,[np.array([i+1.]) for i in range(n)],task=name,source=dict(kind='synthetic'))
 return root

class Sim:
 def __init__(self,spec,i):self.env=ChangingDummyEpisode();self.closed=False
 async def start(self):return dict(source_manifest=runtime_source_manifest(),kind='cpu_dummy',pid=-1)
 async def reset(self,s,seed):
  o,h=self.env.reset(s,seed);return dict(observation=o,restored_state_sha256=h,success=False)
 async def step(self,a,n):return self.env.step(a,n)
 async def close(self):self.closed=True

def run(tmp_path,*,n=5,phase='formal',cell='T02-4-selected',sim_factory=Sim,transport_factory=None,**kwargs):
 root=banks(tmp_path,n);s=prepare(root,mode='cpu_dummy',phase=phase,selected_batch=4,**kwargs)
 out=tmp_path/'out';r=asyncio.run(run_cell(s,root,cell,out,sim_python=sys.executable,sim_factory=sim_factory,transport_factory=transport_factory))
 return r,out,s


def test_formal_task_width_arm_matrix_and_same_state_set(tmp_path):
 root=banks(tmp_path);s=prepare(root,mode='cpu_dummy',phase='formal',selected_batch=4)
 assert len(s['cells'])==27 and len(s['randomized_order'])==27 and validate(s,root)==s
 assert {c['concurrency'] for c in s['cells']}=={1,2,4}
 assert not s['formal_LIBERO_claim_eligible']
 for task in TASKS:assert len(s['banks'][task]['cases'])==5


def test_actual_batch_tail_no_padding_and_next_observation(tmp_path):
 r,out,s=run(tmp_path)
 assert r['status']=='complete' and r['planned_cases']==r['successful_cases']==5
 assert r['batch_sizes']==[1,4] and reaudit(out)['actions_revalidated']==10
 assert all(o['executed_steps']==11 and o['action_chunks']==2 for o in r['outcomes'])
 assert len(list(out.glob('cases/*/observation-*')))==15

@pytest.mark.parametrize('arm',['official','shared_b1'])
def test_reference_arms_are_batch_one_under_concurrency(tmp_path,arm):
 r,_,_=run(tmp_path,cell='T01-4-'+arm);assert r['batch_sizes']==[1] and r['successful_cases']==5


def test_real_independent_cpu_simulator_processes(tmp_path):
 r,out,_=run(tmp_path,n=3,sim_factory=None)
 assert r['status']=='complete' and r['batch_sizes']==[3]
 events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
 assert len({e['evidence']['pid'] for e in events if e['kind']=='simulator_start'})==3


def test_wrong_response_identity_no_actions_or_denominator_loss(tmp_path):
 class Wrong(CPUTransport):
  def predict(self,*args):
   rows=super().predict(*args);rows[0]['metadata']['robot_id']='other';return rows
 r,out,_=run(tmp_path,transport_factory=Wrong)
 assert r['status']=='failed' and r['planned_cases']==5 and r['successful_cases']==0
 assert [o['status'] for o in r['outcomes']]==['error']*4+['not_started']
 assert not list(out.glob('cases/*/action*')) and reaudit(out)['actions_revalidated']==0


def test_step_failure_retains_unknown_and_sibling_success(tmp_path):
 class Bad(Sim):
  async def step(self,a,n):
   if self.env.value==1:raise TimeoutError('lost simulator reply')
   return await super().step(a,n)
 r,out,_=run(tmp_path,sim_factory=Bad)
 assert r['status']=='failed' and r['unknown_step_applications']==[[0,0]]
 assert r['outcomes'][1]['executed_steps']==8
 assert reaudit(out)['unknown_step_applications']==[[0,0]]


def test_already_successful_reset_never_calls_model(tmp_path):
 class Done(Sim):
  async def reset(self,*args):r=await super().reset(*args);r['success']=True;return r
 class NoInference(CPUTransport):
  def predict(self,*args):raise AssertionError('must not infer')
 r,out,_=run(tmp_path,sim_factory=Done,transport_factory=NoInference)
 assert r['successful_cases']==5 and r['batch_sizes']==[] and reaudit(out)['actions_revalidated']==0


def test_step_limit_and_timeout(tmp_path):
 r,out,_=run(tmp_path,max_steps=9)
 assert r['status']=='complete' and r['successful_cases']==0
 assert all(o['executed_steps']==9 for o in r['outcomes'])


def test_total_timeout_keeps_intended_cases(tmp_path):
 class Slow(Sim):
  async def step(self,*args):await asyncio.sleep(5)
 r,out,_=run(tmp_path,sim_factory=Slow,trial_timeout_s=1)
 assert r['status']=='failed' and r['failure']['type']=='TimeoutError' and len(r['unknown_step_applications'])==4
 assert r['outcomes'][-1]['status']=='not_started'


def test_tampered_saved_actions_and_outcome_refused(tmp_path):
 r,out,_=run(tmp_path)
 p=out/'cases/0/action-0.npz';np.savez_compressed(p,**{'action.x':np.zeros((1,8,1),np.float32)})
 with pytest.raises(ValueError,match='evidence changed'):reaudit(out)
 (out/'evidence.json').unlink();seal(out)
 with pytest.raises(ValueError,match='Action array'):reaudit(out)


def test_task_bank_mismatch_and_gpu_synthetic_rejected(tmp_path):
 root=banks(tmp_path);p=root/'T02/initial_states.json';m=json.loads(p.read_text());m['task']=TASKS['T01'];p.write_text(json.dumps(m))
 with pytest.raises(ValueError,match='task mismatch'):prepare(root,mode='cpu_dummy',phase='formal',selected_batch=4)
 with pytest.raises(ValueError,match='fixed 50'):prepare(root,mode='libero_gpu',phase='formal',selected_batch=4)


def test_batch_worker_recovers_admission_error_and_rejects_duplicate():
 w=RegressionBatchWorker(DummyPolicy());ack=w.reset('r','a','e');m=dict(ack,request_id='x',attempt_id='a',component_id='s1',observation_seq=0)
 item=dict(metadata=m,observation=observation(1),seed=42)
 with pytest.raises(ValueError):w.predict([item,item],'selected')
 m2=dict(m,request_id='y',observation_seq=1)
 assert w.predict([dict(item,metadata=m2)],'selected')[0]['action']['action.x'][0,0,0]==1
 with pytest.raises(ValueError):w.predict([dict(item,metadata=m2)],'selected')
