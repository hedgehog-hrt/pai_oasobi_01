"""Bounded device samples: sampled maxima, not a continuous peak guarantee."""
import asyncio,time
from collections import Counter
from .adapter import JsonlSink


def summarize(rows):
    for row in rows:
        if row['finished_ns']<row['started_ns'] or row['stage'] not in ('before','periodic','after'):raise ValueError('Invalid telemetry interval')
        if not isinstance(row['sample'].get('kind'),str):raise ValueError('Missing telemetry kind')
    return dict(samples=len(rows),kinds=dict(Counter(r['sample']['kind'] for r in rows)),
        periodic_samples=sum(r['stage']=='periodic' for r in rows),interval_s=1.,timeout_s=5.,
        continuous_peak_guarantee=False,per_request_attribution=False)


class Recorder:
    def __init__(self,path,fetch):
        self.sink=JsonlSink(path);self.fetch=fetch;self.rows=[];self.task=None
    async def sample(self,stage):
        start=time.monotonic_ns()
        try:sample=await asyncio.wait_for(self.fetch(),5.)
        except Exception as exc:sample=dict(kind='unavailable',error_type=type(exc).__name__)
        row=dict(started_ns=start,finished_ns=time.monotonic_ns(),stage=stage,sample=sample)
        self.rows.append(row);self.sink(row);return sample
    def start(self):
        async def loop():
            while True:
                await asyncio.sleep(1.);await self.sample('periodic')
        self.task=asyncio.create_task(loop())
    async def stop(self):
        if self.task:
            self.task.cancel();await asyncio.gather(self.task,return_exceptions=True)
        try:return await self.sample('after')
        finally:self.sink.close()
