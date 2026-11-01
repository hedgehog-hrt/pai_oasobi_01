import json,math,time
from test_closed_loop import suite,run
from rosa_repro import closed_loop_runtime as runtime

def test_obligation_due_during_terminal_snapshot_is_serviced_before_reset(tmp_path,monkeypatch):
 s=suite(tmp_path,('P2',),warmup_s=.04,duration_s=.5)
 spec=next(x for x in s['plan']['classes'][0]['audit']['components'] if x['role']=='monitor')
 original=runtime.save_observation;delayed=[]
 def save(path,obs,metadata):
  value=original(path,obs,metadata)
  if metadata['actual_steps']==11 and not delayed:
   life=[json.loads(x) for x in (tmp_path/'run/lifecycle.jsonl').read_text().splitlines()]
   epoch=next(x for x in life if x['kind']=='run_started')['epoch_ns']
   base=epoch+spec['phase_ns'];slot=(time.monotonic_ns()-base)//spec['period_ns']+1
   due=base+slot*spec['period_ns'];time.sleep(max(0,(due-time.monotonic_ns())/1e9)+.002)
   delayed.append((metadata['episode_id'],slot,due))
  return value
 monkeypatch.setattr(runtime,'save_observation',save)
 result,_=run(s,tmp_path);assert result['status']=='complete',result
 assert len(delayed)==1
 episode,slot,due=delayed[0]
 requests=[json.loads(x) for x in (tmp_path/'run/requests.jsonl').read_text().splitlines()]
 matches=[r for r in requests if r['component_id']=='monitor' and r['slot_index']==slot]
 assert len(matches)==1,'Terminal snapshot made an already-due monitor obligation disappear'
 row=matches[0];assert row['episode_id']==episode and row['scheduled_ns']==due
 assert due<=row['sent_ns']<=row['finished_ns']<=due+round(spec['deadline_ms']*1e6)
 detail=next(json.loads(x) for x in (tmp_path/'run/request_details.jsonl').read_text().splitlines() if json.loads(x)['request_id']==row['request_id'])
 assert json.loads(detail['context']['execution_history'])['actual_steps']==11
 assert runtime.reaudit_closed(tmp_path/'run')['actions_revalidated']>0

def test_shutdown_does_not_issue_a_future_slot_or_reuse_an_old_episode(tmp_path):
 s=suite(tmp_path,('P2',),warmup_s=.04,duration_s=.5)
 result,_=run(s,tmp_path);assert result['status']=='complete',result
 episodes={(x['robot_id'],x['episode_id']):x for x in map(json.loads,(tmp_path/'run/episodes.jsonl').read_text().splitlines())}
 requests=list(map(json.loads,(tmp_path/'run/requests.jsonl').read_text().splitlines()))
 seen=set()
 for row in requests:
  if row['component_id']!='monitor':continue
  key=(row['robot_id'],row['slot_index']);assert key not in seen;seen.add(key)
  ep=episodes[(row['robot_id'],row['episode_id'])]
  if ep['terminal_ns'] is not None:assert row['scheduled_ns']<=ep['terminal_ns']
  assert ep['start_ns']<=row['scheduled_ns']<=row['sent_ns']<ep['end_ns']
 runtime.reaudit_closed(tmp_path/'run')
