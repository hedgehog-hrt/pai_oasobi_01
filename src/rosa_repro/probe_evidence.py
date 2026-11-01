"""Bounded, single-writer evidence persistence outside the load-generator loop."""
import asyncio


class EvidenceWriter:
    def __init__(self,capacity=64):
        self.queue=asyncio.Queue(maxsize=capacity)
        self.capacity=capacity;self.peak=0;self.errors=[];self.written=0
        self.task=asyncio.create_task(self._run())

    def submit(self,operation):
        self.queue.put_nowait(operation)
        self.peak=max(self.peak,self.queue.qsize())

    async def _run(self):
        while True:
            operation=await self.queue.get()
            try:
                if operation is None:return
                try:
                    await asyncio.to_thread(operation)
                    self.written+=1
                except Exception as exc:self.errors.append(type(exc).__name__)
            finally:self.queue.task_done()

    async def close(self):
        await self.queue.join()
        await self.queue.put(None)
        await self.task
        return dict(policy='bounded_single_writer_thread_v1',capacity=self.capacity,
                    peak_queued=self.peak,written=self.written,errors=self.errors)
