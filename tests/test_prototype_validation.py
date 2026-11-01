import asyncio,os,json
from copy import deepcopy
import pytest
from rosa_repro.prototype_plan import prepare,validate,check_inventory
from rosa_repro.profile_protocol import template
from rosa_repro.execution_plan import DUMMY_FACTORY,check_runtime_inventory
from rosa_repro.dedicated_core import GroupCore
from rosa_repro.dummy import observation
from rosa_repro.pipeline_runtime import run_pipeline,reaudit
from rosa_repro.closed_loop_protocol import protocol


def configs():
    values={}
    for role in ('s1','s2','safety','monitor'):
        c=template(role,'groot' if role=='s1' else 'vllm')
        c.update(mode='cpu_dummy',model_revision='cpu',runtime_fingerprint='cpu',hardware_kind='cpu',streams=4,
                 binding=dict(factory=DUMMY_FACTORY,kwargs={}))
        values[role]=c
    return values


class Local:
    def __init__(self,plan):self.group=GroupCore(plan['workers'],plan['plan_id'])
    async def reset(self,wid,*args):return self.group.reset(wid,*args)
    async def predict(self,wid,*args):return await self.group.predict(wid,*args)
    async def inventory(self):return {wid:dict(self.group.stats(wid),node_id='local',pid=os.getpid(),gpu_ids=[]) for wid in self.group.cores}


def test_shared_models_keep_robot_sessions_and_feed_existing_pipeline_audit(tmp_path):
    plan=prepare(configs(),{'P1':1,'P2':1,'P3':1,'P4':1},frequency_hz=40,action_time_s=.001)
    assert len(plan['workers'])==4 and plan['single_gpu_colocation_extension'] and not plan['paper_model_placement']
    async def go():
        t=Local(plan)
        result=await run_pipeline(plan,t,lambda robot,seq:observation(1),tmp_path/'run',warmup_s=0,duration_s=.18,request_timeout_s=1)
        await t.group.drain()
        inv=await t.inventory();check_runtime_inventory(plan,inv)
        assert t.group.peak==1
        return result,inv
    result,inv=asyncio.run(go());assert result['status']=='complete'
    assert reaudit(tmp_path/'run',tmp_path/'reaudit.json')==json.loads((tmp_path/'run/metrics.json').read_text())
    assert sum(v['backend_plan_inputs'] or 0 for v in inv.values())>0
    assert not result['formal_performance_claim']
    bad=deepcopy(inv);keys=list(bad);bad[keys[1]]['instance_id']=bad[keys[0]]['instance_id']
    with pytest.raises(ValueError,match='alias'):check_inventory(plan,bad)
    bad=deepcopy(inv);bad[keys[1]]['pid']+=1
    with pytest.raises(ValueError,match='physical'):check_inventory(plan,bad)


def test_prototype_is_explicit_and_cannot_be_relabelled_formal():
    values=configs();p1=prepare({'s1':values['s1']},{'P1':4})
    assert p1['paper_model_placement'] and not p1['single_gpu_colocation_extension']
    with pytest.raises(ValueError):protocol(p1,'formal')
    with pytest.raises(ValueError):protocol(p1,'gpu_connection')
    changed=deepcopy(p1);changed['formal_performance_claim']=True
    with pytest.raises(ValueError):validate(changed)
    values['s1']['batch_size']=2
    with pytest.raises(ValueError,match='batch one'):prepare(values,{'P4':4})
    values['s1']['batch_size']=1;values['monitor']['binding']['runtime_env']={'py_executable':'/different/python'}
    with pytest.raises(ValueError,match='Common'):prepare(values,{'P4':1})


def test_gpu_connection_protocol_does_not_weaken_reviewed_r07_requirements():
    # Only construct descriptors here; no fake evidence or GPU invocation.
    values={r:template(r,'groot' if r=='s1' else 'vllm') for r in ('s1','s2','safety','monitor')}
    from rosa_repro.vlm_contract import runtime_fingerprint
    for r,c in values.items():
        c['hardware_kind']='GPU';c['binding']['runtime_env']={'py_executable':'/opt/python'}
        c['binding']['kwargs']['model_path']='/models/'+r
        if r=='s1':c['binding']['kwargs'].update(repo='/repo',stateless_verified=True);c['runtime_fingerprint']='numeric-verified-fixture'
        else:
            cfg=c['binding']['kwargs']['config'];cfg['gpu_memory_utilization']=.2;c['runtime_fingerprint']=runtime_fingerprint(cfg)
    p=prepare(values,{'P4':1});pro=protocol(p,'gpu_connection',warmup_s=10,duration_s=20)
    assert pro['min_warmup_success_per_active_class_component']==1 and pro['min_measure_success_per_active_class_component']==1
    with pytest.raises(ValueError):protocol(p,'formal')
    values['safety']['binding']['kwargs']['config']['gpu_memory_utilization']=.7
    values['safety']['runtime_fingerprint']=runtime_fingerprint(values['safety']['binding']['kwargs']['config'])
    with pytest.raises(ValueError,match='headroom'):prepare(values,{'P4':1})
