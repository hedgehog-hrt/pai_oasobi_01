"""CPU in-process protocol harness. Real Ray verification uses fleet_service."""
import asyncio
import os
from .component_runtime import ComponentCore,AdmissionRejected
from .execution_plan import validate_execution_plan


class LocalFleetTransport:
    def __init__(self,plan):
        validate_execution_plan(plan)
        if plan['mode']!='cpu_dummy':raise ValueError('Local harness is CPU dummy only')
        self.cores={w['worker_id']:ComponentCore(w,plan['plan_id']) for w in plan['workers']}
        self.locks={w:asyncio.Lock() for w in self.cores};self.pending=set()
    async def reset(self,wid,*args):return self.cores[wid].reset(*args)
    async def predict(self,wid,meta,payload):
        c=self.cores[wid]
        try:ticket=c.accept(meta,payload)
        except AdmissionRejected:return dict(c.response({'meta':meta}),status='rejected')
        async def work():
            try:
                async with self.locks[wid]:return (await c.run_batch([ticket]))[0]
            finally:c.finish(ticket)
        task=asyncio.create_task(work());self.pending.add(task);task.add_done_callback(self.pending.discard)
        return await asyncio.shield(task)
    async def inventory(self):
        return {wid:dict(c.stats(),node_id='in-process-harness',pid=os.getpid(),gpu_ids=[],cuda_visible_devices=None)
                for wid,c in self.cores.items()}
    async def drain(self):
        if self.pending:await asyncio.gather(*self.pending,return_exceptions=True)
