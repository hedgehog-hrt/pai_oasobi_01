"""Bounded R06 episodes in independent simulators; retain every intended case and exact action."""
import asyncio,json,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .adapter import check_action_batch_one,check_observation_batch_one,JsonlSink
from .paired import load_state_bank,digest_value
from .regression import save_observation,load_observation,write_json,sha256
from .execution_plan import digest,runtime_source_manifest
from .r06_protocol import TASKS,validate
from .r06_service import RegressionBatchWorker,ZmqBatchTransport
from .simulator_process import ProcessSimulator

class CPUTransport(RegressionBatchWorker):
    def __init__(self):
        from .dummy import DummyPolicy
        super().__init__(DummyPolicy())
    def evidence(self):return dict(kind='synthetic',source_manifest=runtime_source_manifest())
    def close(self):pass


def seal(out):
    files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence.json'}
    write_json(out/'evidence.json',dict(files=files,sha256=digest(files)))


def verify(out):
    e=json.loads((out/'evidence.json').read_text())
    if digest(e['files'])!=e['sha256']:raise ValueError('R06 evidence index changed')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='evidence.json'}
    if actual!=set(e['files']):raise ValueError('R06 evidence coverage differs')
    for name,h in e['files'].items():
        p=(out/name).resolve()
        if not p.is_relative_to(out.resolve()) or sha256(p)!=h:raise ValueError('R06 evidence changed')
    return e


async def run_cell(suite,bank_root,cell_id,output,*,sim_python,sim_repo=None,port=5556,
                   transport_factory=None,sim_factory=None):
    validate(suite,bank_root)
    matches=[c for c in suite['cells'] if c['id']==cell_id]
    if len(matches)!=1:raise ValueError('Unknown R06 cell')
    if suite['mode']=='libero_gpu' and (transport_factory or sim_factory):raise ValueError('GPU fixture injection forbidden')
    cell=matches[0];b=suite['budgets'];manifest,states=load_state_bank(Path(bank_root)/cell['task'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write_json(out/'suite.json',suite)
    write_json(out/'condition.json',dict(suite_sha256=suite['sha256'],cell=cell));source=runtime_source_manifest()
    write_json(out/'source_manifest.json',source);run_id='r06-'+uuid.uuid4().hex
    outcomes=[dict(case_id=c['case_id'],seed=c['seed'],state_sha256=c['state_sha256'],task=cell['task'],
        started=False,status='not_started',success=None,executed_steps=0,action_chunks=0,
        restored_state_sha256=None,initial_observation_sha256=None) for c in manifest['cases']]
    events=[];sink=JsonlSink(out/'events.jsonl');sims={};transport=None;failure=None
    executor=ThreadPoolExecutor(max_workers=1);loop=asyncio.get_running_loop()
    def emit(kind,**kwargs):
        row=dict(kind=kind,time_ns=time.monotonic_ns(),**kwargs);events.append(row);sink(row)
    async def call(fn,*args,**kwargs):
        import functools
        return await loop.run_in_executor(executor,functools.partial(fn,*args,**kwargs))
    expected=suite['server_evidence'] if suite['mode']=='libero_gpu' else dict(kind='synthetic',source_manifest=source)
    def capture(idx,seq,obs):
        save_observation(out/'cases'/str(idx)/f'observation-{seq}',obs,dict(task=cell['task'],case=manifest['cases'][idx],
            arm=cell['arm'],observation_seq=seq,mode=suite['mode']))
    async def body():
        nonlocal transport
        factory=transport_factory or (CPUTransport if suite['mode']=='cpu_dummy' else lambda:ZmqBatchTransport(port,b['call_timeout_s']))
        transport=await call(factory)
        observed=await call(transport.evidence);write_json(out/'server_before.json',observed)
        if observed!=expected:raise ValueError('Server evidence differs from frozen suite')
        for start in range(0,len(outcomes),cell['concurrency']):
            ids=list(range(start,min(start+cell['concurrency'],len(outcomes))))
            spec=dict(kind='cpu_dummy') if suite['mode']=='cpu_dummy' else dict(kind='libero',repo=str(Path(sim_repo).resolve()),
                modality_config=expected['modality_config'],task=TASKS[cell['task']],max_steps=b['max_steps'],execute_steps=b['execute_steps'])
            for i in ids:
                sims[i]=(sim_factory(spec,i) if sim_factory else ProcessSimulator(spec,python=sim_python,
                    stderr_path=out/f'simulator-{i}.stderr',timeout_s=b['call_timeout_s']))
                outcomes[i].update(started=True,status='error')
            starts=await asyncio.gather(*(sims[i].start() for i in ids))
            for i,info in zip(ids,starts):
                if info['source_manifest']!=source:raise ValueError('Simulator source differs')
                emit('simulator_start',index=i,evidence=info)
            resets=await asyncio.gather(*(sims[i].reset(states[manifest['cases'][i]['case_id']],manifest['cases'][i]['seed']) for i in ids))
            active={}
            for i,r in zip(ids,resets):
                o=outcomes[i];obs=r['observation'];check_observation_batch_one(obs);capture(i,0,obs)
                o.update(restored_state_sha256=r['restored_state_sha256'],initial_observation_sha256=digest_value(obs))
                robot=f'env-{i-start}';episode=manifest['cases'][i]['case_id']
                ack=await call(transport.reset,run_id=run_id,robot_id=robot,episode_id=episode)
                if any(ack[k]!=v for k,v in dict(run_id=run_id,robot_id=robot,episode_id=episode).items()) or type(ack['generation']) is not int or ack['generation']<0:
                    raise ValueError('Reset acknowledgement identity differs')
                emit('reset',index=i,restored_state_sha256=o['restored_state_sha256'],observation_sha256=digest_value(obs),success=r['success'],ack=ack)
                if r['success']:o.update(status='completed',success=True,termination_reason='reset_success')
                else:active[i]=dict(observation=obs,ack=ack,seq=0)
            while active:
                order=list(active)
                for offset in range(0,len(order),cell['max_batch_size']):
                    batch=order[offset:offset+cell['max_batch_size']];items=[]
                    for i in batch:
                        a=active[i];metadata=dict(a['ack'],request_id=uuid.uuid4().hex,attempt_id='attempt-001',component_id='s1',observation_seq=a['seq'])
                        item=dict(metadata=metadata,observation=a['observation'],seed=outcomes[i]['seed']+a['seq']);items.append(item)
                        emit('request',index=i,metadata=metadata,seed=item['seed'],observation_sha256=digest_value(item['observation']))
                    replies=await call(transport.predict,items,cell['arm'])
                    if len(replies)!=len(items):raise ValueError('Response row count differs')
                    if len({r['batch_id'] for r in replies})!=1:raise ValueError('Batch IDs differ')
                    # Validate every reply before applying any member of the batch.
                    for i,item,r in zip(batch,items,replies):
                        if (r['metadata']!=item['metadata'] or r['status']!='ok' or r['actual_batch_size']!=len(items)
                            or r['batch_seed']!=items[0]['seed'] or r['requested_seed']!=item['seed']
                            or r['input_sha256']!=digest_value(item['observation'])):raise ValueError('Response attribution differs')
                        check_action_batch_one(r['action'])
                        if min(v.shape[1] for v in r['action'].values())<min(b['execute_steps'],b['max_steps']-outcomes[i]['executed_steps']):
                            raise ValueError('Action horizon shorter than execution stride')
                    limits=[]
                    for i,r in zip(batch,replies):
                        seq=active[i]['seq'];np.savez_compressed(out/'cases'/str(i)/f'action-{seq}.npz',**r['action'])
                        emit('response',index=i,**{k:v for k,v in r.items() if k!='action'},action_sha256=digest_value(r['action']))
                        limit=min(b['execute_steps'],b['max_steps']-outcomes[i]['executed_steps']);limits.append(limit)
                        emit('step_start',index=i,seq=seq,requested_steps=limit,action_sha256=digest_value(r['action']))
                    steps=await asyncio.gather(*(sims[i].step(r['action'],n) for i,r,n in zip(batch,replies,limits)),return_exceptions=True)
                    step_errors=[]
                    for i,r,n in zip(batch,steps,limits):
                        if isinstance(r,BaseException):step_errors.append(r);continue
                        if type(r['steps']) is not int or not 1<=r['steps']<=n:raise ValueError('Invalid actual simulator steps')
                        if type(r['success']) is not bool or type(r['done']) is not bool:raise ValueError('Boolean outcome required')
                        o=outcomes[i];seq=active[i]['seq'];capture(i,seq+1,r['observation'])
                        emit('step_result',index=i,seq=seq,steps=r['steps'],success=r['success'],done=r['done'],observation_sha256=digest_value(r['observation']))
                        o['executed_steps']+=r['steps'];o['action_chunks']+=1
                        if r['success'] or r['done'] or o['executed_steps']>=b['max_steps']:
                            o.update(status='completed',success=r['success'],termination_reason='success' if r['success'] else 'done' if r['done'] else 'step_limit');del active[i]
                        else:active[i].update(observation=r['observation'],seq=seq+1)
                    if step_errors:raise step_errors[0]
            for i in ids:await sims[i].close();del sims[i]
        observed=await call(transport.evidence);write_json(out/'server_after.json',observed)
        if observed!=expected:raise ValueError('Server evidence changed during trial')
    try:
        async with asyncio.timeout(b['trial_timeout_s']):await body()
    except Exception as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    finally:
        cleanup=await asyncio.gather(*(s.close() for s in sims.values()),return_exceptions=True)
        if transport:
            try:await call(transport.close)
            except Exception as exc:cleanup.append(exc)
        executor.shutdown(wait=True,cancel_futures=True)
    cleanup_errors=[str(e) for e in cleanup if isinstance(e,BaseException)]
    if cleanup_errors and not failure:failure=dict(type='CleanupError',message='; '.join(cleanup_errors))
    starts={(r['index'],r['seq']) for r in events if r['kind']=='step_start'}
    ends={(r['index'],r['seq']) for r in events if r['kind']=='step_result'}
    result=dict(schema_version='1.0',suite_sha256=suite['sha256'],cell=cell,mode=suite['mode'],run_id=run_id,
        status='complete' if failure is None else 'failed',failure=failure,outcomes=outcomes,
        planned_cases=len(outcomes),successful_cases=sum(o['success'] is True for o in outcomes),
        completed_cases=sum(o['status']=='completed' for o in outcomes),unknown_step_applications=[list(k) for k in sorted(starts-ends)],
        cleanup_errors=cleanup_errors,model_performance_claim=False,formal_LIBERO_claim=False,
        batch_sizes=sorted({e['actual_batch_size'] for e in events if e['kind']=='response'}))
    write_json(out/'result.json',result);seal(out);reaudit(out);return result


def reaudit(directory):
    out=Path(directory);e=verify(out);read=lambda n:json.loads((out/n).read_text())
    suite=read('suite.json');base=dict(suite);base.pop('sha256')
    if digest(base)!=suite['sha256']:raise ValueError('Suite hash changed')
    result=read('result.json');cell=result['cell'];condition=read('condition.json')
    if read('source_manifest.json')!=suite['source_manifest']:raise ValueError('R06 runtime source differs from suite')
    if result['mode']!=suite['mode'] or result['status'] not in ('complete','failed'):raise ValueError('Invalid result provenance/status')
    if cell not in suite['cells'] or condition!=dict(cell=cell,suite_sha256=suite['sha256']) or result['suite_sha256']!=suite['sha256']:
        raise ValueError('Cell binding differs')
    cases=suite['banks'][cell['task']]['cases'];events=[json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
    if result['planned_cases']!=len(cases) or len(result['outcomes'])!=len(cases):raise ValueError('Case coverage differs')
    if any(a['time_ns']>b['time_ns'] for a,b in zip(events,events[1:])):raise ValueError('Event clock order differs')
    if any(type(e['index']) is not int or not 0<=e['index']<len(cases) for e in events):raise ValueError('Unknown case index')
    requests={};responses={};unknown=[];sizes=set()
    for i,(case,o) in enumerate(zip(cases,result['outcomes'])):
        if any(o[k]!=case[k] for k in ('case_id','seed','state_sha256')) or o['task']!=cell['task']:raise ValueError('Case identity differs')
        if o['status'] not in ('completed','not_started','error') or o['started']!=(o['status']!='not_started'):raise ValueError('Invalid case status')
        ev=[r for r in events if r['index']==i];resets=[r for r in ev if r['kind']=='reset']
        if len(resets)>1 or (not o['started'] and ev):raise ValueError('Repeated/unstarted episode')
        if resets:
            obs,_=load_observation(out/'cases'/str(i)/'observation-0')
            if digest_value(obs)!=resets[0]['observation_sha256'] or o['initial_observation_sha256']!=digest_value(obs) or o['restored_state_sha256']!=resets[0]['restored_state_sha256']:
                raise ValueError('Initial observation/restoration differs')
        steps=[];begins={};seq=0;terminated=bool(resets and resets[0]['success'])
        for r in ev:
            if r['kind'] not in ('simulator_start','reset','request','response','step_start','step_result'):raise ValueError('Unknown R06 event')
            if r['kind']=='simulator_start' and r['evidence']['source_manifest']!=suite['source_manifest']:raise ValueError('Simulator source differs')
            if r['kind']=='request':
                m=r['metadata'];rid=m['request_id']
                if terminated or rid in requests or not resets or any(m[k]!=resets[0]['ack'][k] for k in resets[0]['ack']) or m['observation_seq']!=seq or m['run_id']!=result['run_id']:
                    raise ValueError('Stale/repeated request identity')
                obs,_=load_observation(out/'cases'/str(i)/f'observation-{seq}')
                if digest_value(obs)!=r['observation_sha256'] or r['seed']!=case['seed']+seq:raise ValueError('Request input/seed differs')
                requests[rid]=r
            elif r['kind']=='response':
                rid=r['metadata']['request_id'];req=requests.get(rid)
                if r['status']!='ok' or not req or rid in responses or r['metadata']!=req['metadata'] or r['input_sha256']!=req['observation_sha256'] or r['requested_seed']!=req['seed']:
                    raise ValueError('Response attribution differs')
                with np.load(out/'cases'/str(i)/f'action-{seq}.npz',allow_pickle=False) as data:action=dict(data)
                check_action_batch_one(action)
                if digest_value(action)!=r['action_sha256']:raise ValueError('Action array differs')
                if not 1<=r['actual_batch_size']<=min(cell['max_batch_size'],cell['concurrency']):raise ValueError('Invalid actual batch size')
                responses[rid]=r;sizes.add(r['actual_batch_size'])
            elif r['kind']=='step_start':
                if r['seq']!=seq or seq in begins or not any(x['index']==i and x['metadata']['observation_seq']==seq and x['action_sha256']==r['action_sha256'] for x in responses.values()):
                    raise ValueError('Unmatched/repeated action application')
                if r['requested_steps']!=min(suite['budgets']['execute_steps'],suite['budgets']['max_steps']-sum(x['steps'] for x in steps)):raise ValueError('Execution budget differs')
                begins[seq]=r
            elif r['kind']=='step_result':
                if r['seq']!=seq or seq not in begins or not 1<=r['steps']<=begins[seq]['requested_steps']:raise ValueError('Actual step evidence differs')
                obs,_=load_observation(out/'cases'/str(i)/f'observation-{seq+1}')
                if digest_value(obs)!=r['observation_sha256']:raise ValueError('Next observation differs')
                steps.append(r);seq+=1;terminated=r['success'] or r['done'] or sum(x['steps'] for x in steps)>=suite['budgets']['max_steps']
        unknown.extend([i,s] for s in sorted(set(begins)-{x['seq'] for x in steps}))
        if o['executed_steps']!=sum(x['steps'] for x in steps) or o['action_chunks']!=len(steps):raise ValueError('Episode step summary differs')
        if o['status']=='completed':
            success=steps[-1]['success'] if steps else bool(resets and resets[0]['success'])
            if not terminated or o['success'] is not success:raise ValueError('Completed outcome differs')
        elif o['success'] is not None:raise ValueError('Incomplete episode cannot invent success/failure')
    for batch_id in {r['batch_id'] for r in responses.values()}:
        batch=[r for r in responses.values() if r['batch_id']==batch_id]
        first=min(batch,key=lambda r:r['time_ns'])
        if any(r['actual_batch_size']!=len(batch) or r['batch_seed']!=first['requested_seed'] for r in batch):raise ValueError('Batch membership/seed differs')
    if result['unknown_step_applications']!=unknown or result['batch_sizes']!=sorted(sizes):raise ValueError('Trial summary differs')
    if result['successful_cases']!=sum(o['success'] is True for o in result['outcomes']) or result['completed_cases']!=sum(o['status']=='completed' for o in result['outcomes']):raise ValueError('Denominator differs')
    if result['status']=='complete':
        expected=suite['server_evidence'] if suite['mode']=='libero_gpu' else dict(kind='synthetic',source_manifest=suite['source_manifest'])
        if read('server_before.json')!=expected or read('server_after.json')!=expected:raise ValueError('Observed R06 server runtime changed')
    if result['status']=='complete' and (result['completed_cases']!=len(cases) or unknown or result['failure'] or result['cleanup_errors']):raise ValueError('Incomplete trial marked complete')
    return dict(status=result['status'],cell=cell,planned_cases=len(cases),actions_revalidated=len(responses),
        batch_sizes=sorted(sizes),unknown_step_applications=unknown,evidence_sha256=e['sha256'])
