"""CPU control harness; retiring a handle deliberately permits an old reply."""
import asyncio,os
from .component_runtime import ComponentCore
from .dynamic_control import validate


class LocalControlTransport:
    def __init__(self,spec):
        if spec['blueprint']['mode']!='cpu_dummy':raise ValueError('CPU harness only')
        validate(spec);self.cores={w['worker_id']:ComponentCore(w,spec['blueprint']['plan_id']) for w in spec['workers']};self.locks={w:asyncio.Lock() for w in self.cores};self.pending=set()
    async def worker_stats(self,wid):return dict(self.cores[wid].stats(),node_id='local-control',pid=os.getpid(),gpu_ids=[])
    async def inventory(self):return {wid:await self.worker_stats(wid) for wid in self.cores}
    async def reset(self,wid,*args):return self.cores[wid].reset(*args)
    async def predict(self,wid,meta,payload):
        core=self.cores[wid];ticket=core.accept(meta,payload)
        async def work():
            try:
                async with self.locks[wid]:return (await core.run_batch([ticket]))[0]
            finally:core.finish(ticket)
        t=asyncio.create_task(work());self.pending.add(t);t.add_done_callback(self.pending.discard)
        return await asyncio.shield(t)
    def fail_worker(self,wid):self.cores.pop(wid)
    async def drain(self):
        if self.pending:await asyncio.gather(*self.pending,return_exceptions=True)
