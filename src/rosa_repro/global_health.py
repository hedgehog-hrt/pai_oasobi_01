"""Bounded health-triggered global recovery, for owned model actors only."""
import asyncio


async def watch(supervisor,stop,*,interval_s=1.,timeout_s=5.,max_recoveries=3):
    if not .05<=interval_s<=60 or not .01<=timeout_s<=60 or type(max_recoveries) is not int or not 1<=max_recoveries<=20:raise ValueError('Bounded health policy required')
    recoveries=0
    while not stop.is_set() and not supervisor.closed and not supervisor.blocked:
        current=supervisor.current;epoch=supervisor.epoch
        if current and not supervisor.moving:
            for wid,expected in list(current.inventory.items()):
                if stop.is_set() or supervisor.moving or supervisor.epoch!=epoch:break
                try:
                    found=await asyncio.wait_for(current.transport.worker_stats(wid),timeout_s)
                    if any(found[k]!=expected[k] for k in ('plan_id','instance_id','worker_spec_sha256','runtime_source_sha256')):raise ValueError('Actor identity changed')
                except Exception as exc:
                    if supervisor.moving or supervisor.epoch!=epoch:break
                    supervisor.emit('health_failure',worker_id=wid,epoch=epoch,error_type=type(exc).__name__)
                    if recoveries>=max_recoveries:
                        supervisor.blocked=True;await supervisor._quiesce();supervisor.emit('health_recovery_limit',max_recoveries=max_recoveries);return
                    recoveries+=1
                    result=await supervisor.recover_failure(wid)
                    supervisor.emit('health_recovery_result',count=recoveries,worker_id=wid,failed_epoch=epoch,result=result)
                    break
        try:await asyncio.wait_for(stop.wait(),interval_s)
        except TimeoutError:pass
