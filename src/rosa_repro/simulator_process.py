"""Own simulator subprocess per robot, local IPC, bounded calls and termination.
Allows a separate pinned simulator Python without importing it into the driver.
"""
import argparse,asyncio,os,signal,subprocess,sys,uuid
from pathlib import Path
from multiprocessing.connection import Connection
import socket
from .execution_plan import runtime_source_manifest


class ChangingDummyEpisode:
    def __init__(self,max_steps=11):
        if type(max_steps) is not int or not 1<=max_steps<=1000000:raise ValueError('Invalid CPU fixture step limit')
        self.max_steps=max_steps
    def reset(self,state,seed):
        from .dummy import observation
        from .paired import digest_value
        self.value=float(state[0]);self.steps=0;self.last_success=False
        return observation(self.value),digest_value(state)
    def step(self,action,requested_steps):
        import numpy as np
        from .dummy import observation
        if not np.all(action['action.x']==np.float32(self.value)):raise ValueError('Stale/wrong-robot observation used for action')
        n=min(requested_steps,self.max_steps-self.steps);self.steps+=n;self.value+=n
        return dict(observation=observation(self.value),steps=n,success=self.steps==self.max_steps,done=self.steps==self.max_steps)
    def close(self):pass


def simulator(spec):
    if spec['kind']=='cpu_dummy':return ChangingDummyEpisode(spec.get('max_steps',11))
    if spec['kind']!='libero':raise ValueError('Unsupported simulator')
    from .regression import verify_upstream
    from .libero_bridge import LiberoEpisode
    from types import SimpleNamespace
    verify_upstream(spec['repo']);sys.path.insert(0,spec['repo']);os.chdir(spec['repo'])
    config={k:SimpleNamespace(**v) for k,v in spec['modality_config'].items()}
    if spec.get('collision_guard'):
        from .hybrid_safety import GuardedLiberoEpisode
        factory=GuardedLiberoEpisode;extra={}
        if spec.get('recovery_policy'):
            from .recoverable_sim import RecoverableLiberoEpisode
            factory=RecoverableLiberoEpisode;extra['recovery_policy']=spec['recovery_policy']
        return factory(config,max_steps=spec['max_steps'],execute_steps=spec['execute_steps'],task_name=spec['task'],guard_config=spec['collision_guard'],**extra)
    if spec.get('recovery_policy'):raise ValueError('Retreat requires a collision guard')
    return LiberoEpisode(config,max_steps=spec['max_steps'],execute_steps=spec['execute_steps'],task_name=spec['task'])


def child(fd):
    conn=Connection(fd);env=None
    try:
        while True:
            request=conn.recv();ident=request['id'];op=request['operation'];args=request['args']
            try:
                if op=='start':
                    if env is not None:raise ValueError('Already started')
                    env=simulator(args)
                    from .runtime_attestation import static_environment
                    result=dict(source_manifest=runtime_source_manifest(),environment=static_environment(),kind=args['kind'],pid=os.getpid())
                elif op=='reset':
                    observation,restored=env.reset(args['state'],args['seed'])
                    result=dict(observation=observation,restored_state_sha256=restored,success=bool(getattr(env,'last_success',False)))
                elif op=='step':result=env.step(args['action'],args['requested_steps'])
                elif op=='guard':result=env.guard(args['action'],args['requested_steps'])
                elif op=='retreat':result=env.retreat(args['action'],args['requested_steps'])
                elif op=='close':
                    if env:env.close();env=None
                    conn.send(dict(id=ident,status='ok',result=None));break
                else:raise ValueError('Unsupported simulator operation')
                conn.send(dict(id=ident,status='ok',result=result))
            except Exception as exc:conn.send(dict(id=ident,status='error',error_type=type(exc).__name__,message=str(exc)))
    except EOFError:pass
    finally:
        if env:env.close()
        conn.close()


class ProcessSimulator:
    def __init__(self,spec,*,python,stderr_path,timeout_s):
        self.spec=spec;self.timeout=timeout_s;self.serial=0;self.closed=False;self.lock=asyncio.Lock()
        a,b=socket.socketpair();self.conn=Connection(a.detach());self.log=Path(stderr_path).open('xb')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH',''))
        try:self.proc=subprocess.Popen([str(python),'-m','rosa_repro.simulator_process','--fd',str(b.fileno())],
            pass_fds=(b.fileno(),),stdout=subprocess.DEVNULL,stderr=self.log,env=env)
        except BaseException:self.conn.close();self.log.close();raise
        finally:b.close()
    def abort(self):
        if self.closed:return
        self.closed=True
        for sig in (signal.SIGTERM,signal.SIGKILL):
            try:os.kill(self.proc.pid,sig)
            except ProcessLookupError:pass
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
        self.conn.close();self.log.close()
    def _exchange(self,request):
        self.conn.send(request);r=self.conn.recv()
        if r['id']!=request['id']:raise ValueError('Simulator response identity mismatch')
        if r['status']!='ok':raise RuntimeError(r.get('error_type','SimulatorError')+': '+r.get('message',''))
        return r['result']
    async def call(self,operation,args):
        async with self.lock:
            if self.closed:raise RuntimeError('Simulator already closed')
            task=asyncio.create_task(asyncio.to_thread(self._exchange,dict(id=uuid.uuid4().hex,operation=operation,args=args)))
            try:return await asyncio.wait_for(asyncio.shield(task),self.timeout)
            except BaseException:
                self.abort();await asyncio.gather(task,return_exceptions=True);raise
    async def start(self):return await self.call('start',self.spec)
    async def reset(self,state,seed):return await self.call('reset',dict(state=state,seed=seed))
    async def step(self,action,requested_steps):return await self.call('step',dict(action=action,requested_steps=requested_steps))
    async def guard(self,action,requested_steps):return await self.call('guard',dict(action=action,requested_steps=requested_steps))
    async def retreat(self,action,requested_steps):return await self.call('retreat',dict(action=action,requested_steps=requested_steps))
    async def close(self):
        if self.closed:return
        try:await self.call('close',{})
        finally:self.abort()


class PrefetchedSimulator:
    """Two isolated simulator processes; prepare the next fixed episode off-path.

    Both use the original hard reset. Only the selected process receives actions.
    A prefetched observation cannot be consumed for a different state or seed.
    """
    def __init__(self,spec,*,python,stderr_path,timeout_s):
        self.active=ProcessSimulator(spec,python=python,stderr_path=stderr_path,timeout_s=timeout_s)
        try:self.standby=ProcessSimulator(spec,python=python,stderr_path=str(stderr_path)+'.standby',timeout_s=timeout_s)
        except BaseException:self.active.abort();raise
        self.prepared=None;self.closed=False
    async def start(self):
        evidence,other=await asyncio.gather(self.active.start(),self.standby.start())
        if any(other[k]!=evidence[k] for k in ('source_manifest','kind')):raise ValueError('Standby simulator provenance differs')
        return dict(evidence,reset_prefetch=dict(protocol='two_process_hard_reset_v1',standby_pid=other['pid']))
    def prepare_reset(self,state,seed):
        from .paired import digest_value
        if self.closed or self.prepared is not None:raise ValueError('Reset prefetch already pending or closed')
        self.prepared=(digest_value(state),seed,asyncio.create_task(self.standby.reset(state.copy(),seed)))
    async def reset(self,state,seed):
        from .paired import digest_value
        if self.prepared is None:return await self.active.reset(state,seed)
        h,s,task=self.prepared
        if (h,s)!=(digest_value(state),seed):raise ValueError('Prefetched reset belongs to another state/seed')
        result=await task
        if result['restored_state_sha256']!=h:raise ValueError('Prefetched simulator state differs')
        self.active,self.standby=self.standby,self.active;self.prepared=None
        return result
    async def step(self,*args):return await self.active.step(*args)
    async def guard(self,*args):return await self.active.guard(*args)
    async def retreat(self,*args):return await self.active.retreat(*args)
    async def close(self):
        if self.closed:return
        self.closed=True;failure=None
        if self.prepared:
            task=self.prepared[2]
            if not task.done():task.cancel()
            try:await task
            except asyncio.CancelledError:pass
            except Exception as exc:failure=exc
        results=await asyncio.gather(self.active.close(),self.standby.close(),return_exceptions=True)
        for result in results:
            if isinstance(result,BaseException) and failure is None:failure=result
        if failure:raise failure
    def abort(self):
        self.closed=True
        if self.prepared:self.prepared[2].cancel()
        self.active.abort();self.standby.abort()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fd',type=int,required=True);child(p.parse_args().fd)
