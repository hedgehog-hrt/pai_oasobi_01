import asyncio
import threading
import pytest
from rosa_repro.probe_evidence import EvidenceWriter


def test_slow_writer_does_not_block_loop_and_queue_is_bounded():
    entered=threading.Event();release=threading.Event();thread_ids=[]
    def slow():
        thread_ids.append(threading.get_ident());entered.set()
        assert release.wait(2)
    async def go():
        writer=EvidenceWriter(capacity=2);writer.submit(slow)
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():await asyncio.sleep(.001)
            writer.submit(lambda:None);writer.submit(lambda:None)
            with pytest.raises(asyncio.QueueFull):writer.submit(lambda:None)
            assert thread_ids[0]!=threading.get_ident()
        finally:release.set()
        report=await writer.close()
        assert report['written']==3 and report['peak_queued']==2 and not report['errors']
    asyncio.run(go())


def test_writer_failure_is_reported_and_other_evidence_is_drained():
    saved=[]
    def fail():raise OSError('injected disk failure')
    async def go():
        writer=EvidenceWriter();writer.submit(fail);writer.submit(lambda:saved.append(True))
        report=await writer.close()
        assert report['errors']==['OSError'] and saved==[True]
    asyncio.run(go())
