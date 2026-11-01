"""Owned fleet lifecycle adapters. No cluster lifecycle or Pod operations."""
import asyncio
from .dynamic_control import validate
from .profile_adoption import validate_fleet_adoptions


class LocalManager:
    def __init__(self):self.live=None;self.pending=None;self.max_owned=0;self.history=[]
    async def preflight(self,plan):
        if plan['mode']!='cpu_dummy':raise ValueError('Local manager refuses GPU')
        validate_fleet_adoptions(plan,None)
    async def deploy(self,spec):
        from .control_local import LocalControlTransport
        validate(spec)
        if self.live or self.pending:raise ValueError('Previous leases not released')
        self.live=LocalControlTransport(spec);self.max_owned=max(self.max_owned,len(spec['workers']));self.history.append(('deploy',len(spec['workers'])));return self.live
    async def release(self):
        if self.live:
            await self.live.drain();self.live.cores.clear();self.live=None;self.history.append(('release',0))


class RayManager:
    def __init__(self,adoptions=None):self.live=None;self.pending=None;self.adoptions=adoptions
    def selected(self,plan):
        if plan['mode']=='cpu_dummy':return None
        if not self.adoptions:raise ValueError('GPU catalog requires reviewed adoptions')
        needed={w['profile_id'] for w in plan['workers']}
        return dict(evidence_root=self.adoptions['evidence_root'],adoptions={p:self.adoptions['adoptions'][p] for p in needed})
    async def preflight(self,plan):validate_fleet_adoptions(plan,self.selected(plan))
    async def deploy(self,spec):
        from .fleet_service import deploy_control
        if self.live or self.pending:raise ValueError('Previous fleet operation unresolved')
        async def work():
            try:self.live=await asyncio.to_thread(deploy_control,spec,adoptions=self.selected(spec['blueprint']));return self.live
            finally:self.pending=None
        self.pending=asyncio.create_task(work());return await asyncio.shield(self.pending)
    async def release(self):
        # A timed-out deployment may still be creating actors. Wait for its
        # actual completion before deciding what belongs to this manager.
        if self.pending:
            try:await asyncio.shield(self.pending)
            except Exception:pass
        if self.live:
            t=self.live
            await asyncio.to_thread(t.close);self.live=None
