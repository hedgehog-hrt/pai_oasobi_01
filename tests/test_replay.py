import asyncio
import json
import time
from types import SimpleNamespace

from rosa_repro.adapter import BatchOneWorker
from rosa_repro.dummy import DummyPolicy,observation
from rosa_repro.replay import run_trial


class Reply:
    def __init__(self,value):self.value=value
    def result(self,timeout_s=None):return self.value
    def __await__(self):
        async def value():return self.value
        return value().__await__()


class Method:
    def __init__(self,fn):self.fn=fn
    def remote(self,*a,**kw):return Reply(self.fn(*a,**kw))


def handle(delay=0,fail=False):
    class Policy(DummyPolicy):
        def get_action(self,obs):
            time.sleep(delay)
            if fail:raise RuntimeError('fixture failure')
            return super().get_action(obs)
    w=BatchOneWorker(Policy(),stateless_verified=True)
    return SimpleNamespace(reset=Method(w.reset),predict=Method(w.predict),
        stats=Method(lambda:dict(execution_kind='cpu_dummy')))


def condition(mode='closed_loop'):
    return dict(case_id='test-case',mode=mode,execution_kind='cpu_dummy',phase='cpu_smoke',batch_size=1,
        robots=2,deadline_ms=200,offered_hz=100,seed=42,action_time_s=.001,rate_cap_hz=None,
        protocol=dict(warmup_min_s=.01,warmup_min_success=1,warmup_max_s=.15,
            measure_min_s=.06,measure_min_success=3,measure_max_s=.2,request_timeout_s=1.))


def test_closed_loop_records_virtual_chunks_and_warmup_exclusion(tmp_path):
    r=asyncio.run(run_trial(handle(),[observation(1)],condition(),tmp_path))
    assert r['status']=='complete' and r['metrics']['raw_action_chunks']>0
    assert r['action_semantics']=='virtual_timed_chunks' and not r['profile_export_allowed']
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert any(e.get('warmup') is True for e in events)
    assert r['metrics']['requests']<sum(e['record_type']=='request' for e in events)


def test_warmup_failure_does_not_start_measurement(tmp_path):
    r=asyncio.run(run_trial(handle(fail=True),[observation(1)],condition(),tmp_path))
    assert r['status']=='warmup_failed' and 'measurement' not in r and r['errors']


def test_overload_rejections_stay_in_denominator_and_no_profile_actions(tmp_path):
    c=condition('open_loop');c['offered_hz']=1000
    r=asyncio.run(run_trial(handle(delay=.02),[observation(1)],c,tmp_path))
    assert r['metrics']['request_status_counts']['rejected']>0
    assert r['metrics']['raw_action_chunks']==0 and r['metrics']['request_deadline_meet_rate']<1


def test_measurement_shortfall_explicitly_insufficient(tmp_path):
    c=condition();c['protocol']['measure_min_success']=100000
    r=asyncio.run(run_trial(handle(),[observation(1)],c,tmp_path))
    assert r['status']=='insufficient_or_failed' and not r['measurement']['requirements_met']
