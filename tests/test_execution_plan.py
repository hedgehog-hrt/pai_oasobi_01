from copy import deepcopy
import asyncio
import pytest
from rosa_repro.execution_plan import compile_execution_plan,validate_execution_plan,check_runtime_inventory,DUMMY_FACTORY
from rosa_repro.local_fleet import LocalFleetTransport
from runtime_helpers import plan_for


def test_plan_preserves_heterogeneous_routes_rates_batches_and_full_worker_resources():
    p=plan_for(('P1','P2','P3','P4'));validate_execution_plan(p)
    assert [len(c['robots'][0]['routes']) for c in p['classes']]==[1,2,3,4]
    assert len(p['workers'])==4 and all(w['resources']==dict(num_cpus=1,num_gpus=0) for w in p['workers'])
    assert len({r['robot_id'] for c in p['classes'] for r in c['robots']})==4
    x=p['inputs'];bindings={c:dict(factory='user_backend:factory',kwargs={}) for c in x['roles']}
    g=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings=bindings)
    assert all(w['resources']==dict(num_cpus=1,num_gpus=1) for w in g['workers'])
    validate_execution_plan(g)


@pytest.mark.parametrize('field',['frequency','route','resource','profile','plan_id'])
def test_modified_plan_refused(field):
    p=plan_for()
    if field=='frequency':p['classes'][0]['frequency_hz']=99
    elif field=='route':p['classes'][0]['robots'][0]['routes']['s1']='wrong'
    elif field=='resource':p['workers'][0]['resources']['num_gpus']=.5
    elif field=='profile':p['inputs']['schedule']['placements'][0]['mean_ms']=0
    else:p['plan_id']='bad'
    with pytest.raises(ValueError):validate_execution_plan(p)


def test_unknown_schedule_role_and_dummy_gpu_binding_refused():
    p=plan_for();x=p['inputs']
    s=deepcopy(x['schedule']);s['status']='unknown'
    with pytest.raises(ValueError,match='feasible'):compile_execution_plan(x['fleet'],s,x['roles'])
    roles=deepcopy(x['roles']);roles['monitor']='s1'
    with pytest.raises(ValueError,match='Role'):compile_execution_plan(x['fleet'],x['schedule'],roles)
    with pytest.raises(ValueError,match='mode'):compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',bindings=x['bindings'])


def test_worker_python_binding_is_preserved_and_digest_protected():
    p=plan_for();x=p['inputs'];x['bindings']['monitor']['runtime_env']={'py_executable':'/opt/rosa-vlm/bin/python'}
    updated=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],bindings=x['bindings'])
    validate_execution_plan(updated)
    w=next(w for w in updated['workers'] if w['role']=='monitor')
    assert w['backend']['runtime_env']=={'py_executable':'/opt/rosa-vlm/bin/python'}
    w['backend']['runtime_env']['py_executable']='/opt/another/bin/python'
    with pytest.raises(ValueError):validate_execution_plan(updated)


@pytest.mark.parametrize('env',[{},None,{'pip':['package']},{'py_executable':'python'},
    {'py_executable':'/opt/python -m launcher'},{'py_executable':'/opt/python;touch /tmp/x'},
    {'py_executable':'/opt/python','env_vars':{}}])
def test_worker_python_binding_rejects_unbounded_runtime_changes(env):
    x=plan_for()['inputs'];x['bindings']['monitor']['runtime_env']=env
    with pytest.raises(ValueError,match='runtime_env'):
        compile_execution_plan(x['fleet'],x['schedule'],x['roles'],bindings=x['bindings'])


def test_runtime_inventory_checks_real_gpu_lease_uniqueness_and_worker_identity():
    p=plan_for();t=LocalFleetTransport(p);inventory=asyncio.run(t.inventory())
    assert check_runtime_inventory(p,inventory)['unique_gpu_leases']==0
    keys=list(inventory);wrong=deepcopy(inventory);wrong[keys[0]]['instance_id']=wrong[keys[1]]['instance_id']
    with pytest.raises(ValueError,match='alias'):check_runtime_inventory(p,wrong)
    x=p['inputs'];g=compile_execution_plan(x['fleet'],x['schedule'],x['roles'],mode='gpu',
        bindings={c:dict(factory='user_backend:factory',kwargs={}) for c in x['roles']})
    from rosa_repro.execution_plan import digest
    for w in g['workers']:
        r=inventory[w['worker_id']];r.update(plan_id=g['plan_id'],worker_spec_sha256=digest(w),execution_kind='external_model',gpu_ids=['0'],
            gpu_properties=dict(name=w['hardware_kind'],memory_mib=24000,uuid='synthetic'))
    with pytest.raises(ValueError,match='multiple'):check_runtime_inventory(g,inventory)
    for i,w in enumerate(g['workers']):inventory[w['worker_id']]['gpu_ids']=[str(i)]
    assert check_runtime_inventory(g,inventory)['unique_gpu_leases']==4


@pytest.mark.parametrize('bad',['kind','memory','nan','lease','visible'])
def test_gpu_preflight_rejects_incompatible_resources_without_loading_model(bad):
    from rosa_repro.execution_plan import check_gpu_device
    w=plan_for()['workers'][0];props=dict(name=w['hardware_kind'],memory_mib=24000)
    ids=['0'];visible=1
    if bad=='kind':props['name']='another GPU'
    elif bad=='memory':props['memory_mib']=1
    elif bad=='nan':props['memory_mib']=float('nan')
    elif bad=='lease':ids=[]
    else:visible=2
    with pytest.raises(ValueError):check_gpu_device(w,ids,props,cuda_available=True,visible_count=visible)
