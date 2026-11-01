"""Simulation-only prospective collision guard for the pinned LIBERO T01 task.

The visual VLM checks people and usable views. This independent guard checks the
actual proposed action in a twin simulator, at every 2 ms physics substep. A
clear preview issues a one-use action/state-bound lease; it never moves live.
"""
from copy import deepcopy, copy
import math
import time
import uuid
import numpy as np
from .paired import digest_value

PROTOCOL='libero_t01_shadow_v1'
TASK='pick_up_the_alphabet_soup_and_place_it_in_the_basket'


def config(clearance_m=.002):
    if type(clearance_m) not in (int,float) or not math.isfinite(clearance_m) or not .001<=clearance_m<=.02:
        raise ValueError('Explicit 1–20 mm geometric clearance required')
    return dict(protocol=PROTOCOL,task=TASK,clearance_m=float(clearance_m),max_preview_ms=2500.,max_physics_preview_ms=500.,max_recovery_ms=2000.,simulation_only=True,
        allowed_contacts=['gripper-target','target-basket','target-floor-support','robot-base-floor-mount',
                          'same-rigid-body','adjacent-robot-links','gripper-internal','model-excluded-body-pairs'],
        scope='robot-world, robot self, and target-obstacle collision geometry; no physical-robot safety certification')


def validate_config(value):
    if value!=config(value.get('clearance_m')):raise ValueError('Hybrid safety contract changed')
    return value


def category(name):
    if name.startswith('gripper0_'):return 'gripper'
    if name.startswith('robot0_'):return 'robot'
    if name.startswith('alphabet_soup_1'):return 'target'
    if name.startswith('basket_1'):return 'basket'
    if name=='floor':return 'floor'
    return 'obstacle'


def monitored_pair(a,b,*,same_rigid=False,adjacent=False,excluded=False):
    """Only task-required and mechanical contacts are exempted."""
    ca,cb=category(a),category(b);pair={ca,cb}
    if same_rigid or excluded:return False
    if pair<={'robot','gripper'}:
        return not adjacent and pair!={'gripper'}
    if pair=={'gripper','target'} or pair in ({'target','basket'},{'target','floor'}):return False
    if (a=='robot0_link0_collision' and cb=='floor') or (b=='robot0_link0_collision' and ca=='floor'):return False
    return bool(pair & {'robot','gripper','target'})


class CollisionDetected(Exception):
    def __init__(self,evidence):self.evidence=evidence;super().__init__('Predicted forbidden proximity/contact')


class GeometryChecker:
    def __init__(self,sim,settings):
        import mujoco
        self.mujoco=mujoco;self.sim=sim;self.settings=validate_config(settings);self.margin=settings['clearance_m']
        m=sim.model;self.model=m._model;self.source_data=sim.data._data;self.data=None
        if not hasattr(mujoco,'mj_geomDistance'):raise RuntimeError('MuJoCo distance API unavailable')
        names=[m.geom_id2name(i) or f'unnamed_geom_{i}' for i in range(m.ngeom)]
        active=[i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        if not all(any(category(names[i])==c for i in active) for c in ('robot','gripper','target','basket','floor')):
            raise ValueError('Expected pinned task collision geometry missing')
        excluded=set(int(x) for x in self.model.exclude_signature)
        pairs=[]
        for index,a in enumerate(active):
            for b in active[index+1:]:
                if not (int(m.geom_contype[a]) & int(m.geom_conaffinity[b]) or int(m.geom_contype[b]) & int(m.geom_conaffinity[a])):continue
                ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b]);wa,wb=int(m.body_weldid[ba]),int(m.body_weldid[bb])
                pa=int(m.body_weldid[m.body_parentid[wa]]);pb=int(m.body_weldid[m.body_parentid[wb]])
                adjacent=(wa==pb or wb==pa)
                signature=(min(ba,bb)<<16)+max(ba,bb)
                if not monitored_pair(names[a],names[b],same_rigid=wa==wb,adjacent=adjacent,excluded=signature in excluded):continue
                if int(m.geom_type[a]) not in (0,2,3,4,5,6,7) or int(m.geom_type[b]) not in (0,2,3,4,5,6,7):
                    raise ValueError('Unsupported distance geometry; cannot certify preview')
                pairs.append((a,b))
        if not pairs:raise ValueError('Empty collision guard')
        self.pairs=np.asarray(pairs,dtype=np.int32);self.names=names
        self.radii=np.asarray(m.geom_rbound);self.planes=np.asarray(m.geom_type)==0
        self.monitored=set(pairs);self.probes=0;self.steps=0;self.closest=None

    def check(self):
        # Position/contact refresh mutates MuJoCo caches. Never run it on the
        # forecasting simulator: use the native independent MjData copy instead.
        self.data=copy(self.source_data)
        self.mujoco.mj_fwdPosition(self.model,self.data)
        d=self.data
        for contact in d.contact[:d.ncon]:
            a,b=sorted((int(contact.geom1),int(contact.geom2)))
            if (a,b) in self.monitored:
                distance=float(contact.dist)
                if not math.isfinite(distance):raise ValueError('Nonfinite contact distance')
                if distance<=self.margin:raise CollisionDetected(dict(kind='contact',geoms=[self.names[a],self.names[b]],distance_m=distance))
        a,b=self.pairs.T;pos=np.asarray(d.geom_xpos)
        near=self.planes[a] | self.planes[b] | (np.linalg.norm(pos[a]-pos[b],axis=1)<=self.radii[a]+self.radii[b]+self.margin)
        for ga,gb in self.pairs[near]:
            distance=float(self.mujoco.mj_geomDistance(self.model,self.data,int(ga),int(gb),self.margin,None));self.probes+=1
            if not math.isfinite(distance):raise ValueError('Nonfinite geometry distance')
            if self.closest is None or distance<self.closest:self.closest=distance
            if distance<self.margin:
                raise CollisionDetected(dict(kind='distance',geoms=[self.names[ga],self.names[gb]],distance_m=distance))

    def evidence(self):
        return dict(mujoco_version=self.mujoco.__version__,checked_physics_substeps=self.steps,
            physics_timestep_s=float(self.sim.model.opt.timestep),distance_queries=self.probes,
            monitored_geom_pairs=len(self.pairs),minimum_distance_m=self.closest,
            minimum_distance_clipped_at_m=self.margin)


class GuardedLiberoEpisode:
    def __init__(self,modality_config,max_steps=720,execute_steps=8,task_name=TASK,guard_config=None):
        from .libero_bridge import LiberoEpisode
        self.settings=validate_config(guard_config)
        if task_name!=TASK:raise ValueError('Hybrid geometry contract is task-specific')
        self.kwargs=dict(modality_config=modality_config,max_steps=max_steps,execute_steps=execute_steps,task_name=task_name)
        self.live=LiberoEpisode(**self.kwargs);self.shadow=None;self.lease=None;self.history=[];self.dirty=True

    def state(self,env):return np.asarray(env.base._env.get_sim_state()).copy()

    @staticmethod
    def match(a,b):
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(a,b):
            raise ValueError('Twin/live simulator state differs')

    def cameras(self,active):
        for name,observable in self.shadow.base._env.env._observables.items():
            if name.endswith('_image'):observable.set_active(active)

    def reset(self,state,seed):
        self.initial_state=np.asarray(state).copy();self.seed=seed;self.history=[];self.lease=None;self.dirty=True
        result=self.live.reset(state,seed);self.last_success=self.live.last_success
        self.synchronize();return result

    def synchronize(self):
        from .libero_bridge import LiberoEpisode
        if self.shadow is None:
            self.shadow=LiberoEpisode(**self.kwargs)

        if self.dirty:
            self.cameras(False)
            raw=self.shadow.base._env;raw.seed(self.seed);raw.reset();raw.set_init_state(self.initial_state);self.cameras(False)
            for action,n in self.history:self.preview_steps(action,n)
            self.match(self.state(self.live),self.state(self.shadow));self.dirty=False

    def preview_steps(self,action,n):
        from gr00t.eval.sim.LIBERO.libero_env import normalize_gripper_action,invert_gripper_action
        raw=self.shadow.base._env;done=False;success=False;count=0
        for i in range(n):
            vector=np.concatenate([action['action.'+k][0,i] for k in ('x','y','z','roll','pitch','yaw','gripper')])
            vector=invert_gripper_action(normalize_gripper_action(vector))
            _,_,done,_=raw.step(vector);count+=1;success=success or bool(raw.check_success())
            if done:break
        return dict(steps=count,success=success,done=bool(done))

    def guard(self,action,requested_steps):
        started=time.monotonic_ns();self.lease=None
        if type(requested_steps) is not int or not 1<=requested_steps<=self.live.contract.n_action_steps:
            raise ValueError('Invalid guard horizon')
        before=self.state(self.live);action_hash=digest_value(action)
        evidence=dict(protocol=PROTOCOL,config_sha256=digest_value(self.settings),action_sha256=action_hash,
            state_before_sha256=digest_value(before),requested_steps=requested_steps,live_steps_executed=0,
            status='unknown',lease_id=None)
        sim=None;original=None;checker=None
        try:
            self.synchronize();self.match(before,self.state(self.shadow))
            physics_start=time.monotonic_ns();evidence['recovery_ms']=(physics_start-started)/1e6
            if evidence['recovery_ms']>self.settings['max_recovery_ms']:raise TimeoutError('Twin recovery budget exceeded')
            sim=self.shadow.base._env.env.sim;checker=GeometryChecker(sim,self.settings)
            checker.check();original=sim.step
            def checked_step(*args,**kwargs):
                original(*args,**kwargs);checker.steps+=1;checker.check()
                if (time.monotonic_ns()-physics_start)/1e6>self.settings['max_physics_preview_ms']:raise TimeoutError('Collision preview budget exceeded')
            sim.step=checked_step;self.dirty=True
            preview=self.preview_steps(action,requested_steps)
            after=self.state(self.shadow)
            evidence.update(status='clear',lease_id=uuid.uuid4().hex,predicted_state_after_sha256=digest_value(after),
                predicted_steps=preview['steps'],predicted_success=preview['success'])
            self.lease=dict(action_sha256=action_hash,state_before=before,state_after=after,requested_steps=requested_steps,
                preview=preview,evidence=deepcopy(evidence))
        except CollisionDetected as exc:evidence.update(status='blocked',violation=exc.evidence)
        except Exception as exc:evidence.update(status='unknown',error_type=type(exc).__name__,error=str(exc))
        finally:
            if sim is not None and original is not None:sim.step=original
            if checker:evidence.update(checker.evidence())
            self.match(before,self.state(self.live))
            evidence['elapsed_ms']=(time.monotonic_ns()-started)/1e6
            evidence['physics_preview_ms']=evidence['elapsed_ms']-evidence.get('recovery_ms',evidence['elapsed_ms'])
            evidence['timely']=(evidence['elapsed_ms']<=self.settings['max_preview_ms'] and evidence.get('recovery_ms',float('inf'))<=self.settings['max_recovery_ms'] and evidence['physics_preview_ms']<=self.settings['max_physics_preview_ms'])
            if not evidence['timely']:
                evidence.update(status='unknown',lease_id=None,error_type='PreviewDeadlineExceeded');self.lease=None
        return evidence

    def step(self,action,requested_steps):
        lease=self.lease;self.lease=None
        if lease is None or lease['action_sha256']!=digest_value(action) or lease['requested_steps']!=requested_steps:
            raise ValueError('No matching unused geometric safety lease')
        self.match(lease['state_before'],self.state(self.live))
        result=self.live.step(action,requested_steps)
        self.match(lease['state_after'],self.state(self.live))
        if result['steps']!=lease['preview']['steps'] or result['success']!=lease['preview']['success']:
            raise ValueError('Twin/live outcomes differ')
        self.history.append((deepcopy(action),result['steps']));self.dirty=False
        result['collision_guard']=dict(lease_id=lease['evidence']['lease_id'],action_sha256=lease['action_sha256'],
            state_after_sha256=digest_value(self.state(self.live)),twin_match=True)
        return result

    def close(self):
        try:self.live.close()
        finally:
            if self.shadow is not None:self.shadow.close()


def valid_preview(e,settings,action_hash,steps):
    return (isinstance(e,dict) and e.get('status')=='clear' and e.get('protocol')==PROTOCOL and
        e.get('config_sha256')==digest_value(settings) and e.get('action_sha256')==action_hash and
        e.get('requested_steps')==steps and e.get('live_steps_executed')==0 and
        e.get('timely') is True and type(e.get('elapsed_ms')) in (int,float) and
        0<=e['elapsed_ms']<=settings['max_preview_ms'] and
        0<=e.get('recovery_ms',float('inf'))<=settings['max_recovery_ms'] and
        0<=e.get('physics_preview_ms',float('inf'))<=settings['max_physics_preview_ms'] and
        isinstance(e.get('lease_id'),str) and len(e['lease_id'])==32 and
        type(e.get('predicted_steps')) is int and 1<=e['predicted_steps']<=steps and
        type(e.get('checked_physics_substeps')) is int and e['checked_physics_substeps']>0 and
        e.get('physics_timestep_s')==.002 and e.get('monitored_geom_pairs',0)>0 and
        all(isinstance(e.get(k),str) and len(e[k])==64 for k in ('state_before_sha256','predicted_state_after_sha256')))


def valid_receipt(receipt,preview,steps):
    return isinstance(receipt,dict) and receipt==dict(lease_id=preview['lease_id'],action_sha256=preview['action_sha256'],
        state_after_sha256=preview['predicted_state_after_sha256'],twin_match=True) and steps==preview['predicted_steps']


def audit_guard(plan,records,intents,details):
    """Reject missing, stale, duplicated or unbound authorization for applied actions."""
    settings=validate_config(plan['collision_guard']);used=set()
    guarded={r['robot_id']:next(m for m in c['audit']['components'] if m['role']=='safety')
        for c in plan['classes'] if any(m['role']=='safety' for m in c['audit']['components']) for r in c['robots']}
    def index(kind):
        rows=[x for x in intents if x['kind']==kind];d={x['request_id']:x for x in rows}
        if len(d)!=len(rows):raise ValueError('Duplicate hybrid safety evidence')
        return d
    previews=index('collision_guard_preview');receipts=index('collision_guard_applied');gates=index('hybrid_safety_gate');tensors=index('action_tensor')
    requests={x['request_id']:x for x in records if x['record_type']=='request'}
    for a in (x for x in records if x['record_type']=='action' and x['robot_id'] in guarded):
        rid=a['request_id'];gate=gates.get(rid)
        if gate is None or gate['stopped']!=a['safety_stopped'] or gate['checked_at_ns']!=a['start_ns']:
            raise ValueError('Hybrid decision gate missing or differs')
        if not a['applied']:continue
        comp=guarded[a['robot_id']];same=lambda r:(r['robot_id'],r['episode_id'])==(a['robot_id'],a['episode_id'])
        candidates=[r for r in requests.values() if same(r) and r['component_id']==comp['id'] and r['finished_ns']<=a['start_ns']]
        visual=max(candidates,key=lambda r:r['finished_ns']) if candidates else None
        if visual is None or visual['request_id']!=gate['visual_request_id'] or visual['status']!='ok' or not visual['valid_output']:
            raise ValueError('Missing latest visual authorization')
        d=details[visual['request_id']]
        from .vlm_contract import parse_generation
        evidence=d.get('backend_evidence') or {}
        parsed=parse_generation('safety',evidence.get('raw_generation',''),finish_reason=evidence.get('finish_reason'),format='safety_visual_guard_v1')
        if parsed!=d['semantic_output'] or parsed['stop'] or digest_value(parsed)!=d['output_sha256']:
            raise ValueError('Visual authorization output differs')
        deadline=visual['scheduled_ns']+round(comp['deadline_ms']*1e6)
        if visual['finished_ns']>deadline or a['start_ns']>=deadline+comp['period_ns']:raise ValueError('Expired visual authorization')
        row=previews.get(rid);tensor=tensors.get(rid);receipt=receipts.get(rid)
        if row is None or tensor is None or receipt is None:raise ValueError('Geometric authorization missing')
        e=row['evidence']
        if not valid_preview(e,settings,details[rid]['output_sha256'],tensor['requested_steps']) or not requests[rid]['finished_ns']<=row['start_ns']<=row['end_ns']<=a['start_ns']:
            raise ValueError('Geometric authorization invalid or out of order')
        if gate['cause']!='clear' or not valid_receipt(receipt['receipt'],e,a['executed_steps']) or e['lease_id'] in used:
            raise ValueError('Geometric authorization receipt differs or lease reused')
        used.add(e['lease_id'])
    return dict(applied_guard_leases=len(used),simulation_only=True)
