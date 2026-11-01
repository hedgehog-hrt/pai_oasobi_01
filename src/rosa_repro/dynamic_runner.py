"""Live periodic/S1/S2 driver for the global controller and bounded simulator adapters."""
import asyncio,time
from copy import deepcopy
from .regression import write_json,save_observation
from .paired import digest_value
from .adapter import JsonlSink
from .regression import sha256


async def run(supervisor,simulators,initial_states,changes,*,warmup_s,duration_s,output,sim_timeout_s=10.,max_requests=10000,max_capture_bytes=1024**3,evaluation=None):
    """Changes are (offset seconds from window start, async callback) pairs.
    Callbacks are predeclared by the caller; the CLI uses a finite JSON grammar.
    """
    from pathlib import Path
    out=Path(output);out.mkdir(parents=True,exist_ok=False);states={};jobs=[];requests=0;errors=[];stop=asyncio.Event();capture_bytes=0
    request_log=JsonlSink(out/'request_observations.jsonl');action_log=JsonlSink(out/'actions.jsonl')
    if set(simulators)!=set(initial_states):raise ValueError('Every simulator needs a saved initial state/seed')
    if type(max_requests) is not int or not 1<=max_requests<=100000:raise ValueError('Bounded request budget required')
    if type(max_capture_bytes) is not int or not 1<=max_capture_bytes<=100*1024**3:raise ValueError('Bounded capture budget required')
    if any(type(offset) not in (int,float) or not 0<=offset<duration_s or not callable(callback) for offset,callback in changes):raise ValueError('Changes must fall inside measurement window')
    roles=supervisor.spec['catalog']['roles'];class_by_robot={r:cid for cid,rs in supervisor.spec['initial_population'].items() for r in rs}
    initial_robots=deepcopy(class_by_robot);observed=[];gate=None;window_configured_ns=None
    if evaluation is not None:
        from .dynamic_protocol import validate,sample_counts
        validate(evaluation,supervisor.spec,warmup_s,duration_s)
    # Planned arrivals may not be active initially; their class is declared in initial_states.
    for rid,s in initial_states.items():class_by_robot[rid]=s['class_id']
    async def call(coro):return await asyncio.wait_for(coro,sim_timeout_s)
    try:
        evidence={}
        for rid,sim in simulators.items():
            e=await call(sim.start());evidence[rid]=e
            from .execution_plan import runtime_source_manifest
            if e['source_manifest']!=runtime_source_manifest() or e['kind']!=('cpu_dummy' if supervisor.spec['catalog']['mode']=='cpu_dummy' else 'libero'):raise ValueError('Simulator source/kind differs')
            s=initial_states[rid];reset=await call(sim.reset(s['state'],s['seed']))
            if reset['restored_state_sha256']!=digest_value(s['state']):raise ValueError('Initial simulator state differs')
            states[rid]=dict(observation=reset['observation'],version=0,done=False,steps=0,success=False)
        write_json(out/'simulators.json',evidence)
        write_json(out/'initial_states.json',{rid:dict(class_id=s['class_id'],seed=s['seed'],state_sha256=digest_value(s['state'])) for rid,s in initial_states.items()})
        def capture(rid):
            nonlocal capture_bytes
            s=states[rid];folder=out/'observations'/str(list(states).index(rid))/str(s['version'])
            # Bound the uncompressed input before writing, then count actual artifacts.
            def size(v):
                if isinstance(v,dict):return sum(size(x) for x in v.values())
                if isinstance(v,(list,tuple)):return sum(size(x) for x in v)
                return getattr(v,'nbytes',len(str(v).encode()))
            if capture_bytes+size(s['observation'])>max_capture_bytes:raise ValueError('Dynamic capture budget exhausted')
            save_observation(folder,s['observation'],dict(purpose='dynamic_live_run',robot_id=rid,step=s['steps'],execution_kind=supervisor.spec['catalog']['mode']))
            capture_bytes+=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            if capture_bytes>max_capture_bytes:raise ValueError('Dynamic capture budget exhausted')
        for rid in states:capture(rid)
        start,end=supervisor.set_window(warmup_s,duration_s);epoch=supervisor.rows[0]['time_ns'];window_configured_ns=start-round(warmup_s*1e9)
        def admit():
            nonlocal gate
            if evaluation is None or time.monotonic_ns()<start:return
            if gate is None:
                gate=dict(sample_counts(supervisor.spec,initial_robots,observed,start=window_configured_ns,end=start,minimum=evaluation['warmup_min_success']),decision_ns=time.monotonic_ns())
                write_json(out/'warmup_gate.json',gate)
            if not gate['requirements_met']:raise RuntimeError('DynamicWarmupShortfall')
        async def sleep_until(ns):
            delay=max(0,(ns-time.monotonic_ns())/1e9)
            try:await asyncio.wait_for(stop.wait(),delay)
            except TimeoutError:pass
        async def request(rid,mid,**kw):
            nonlocal requests
            admit()
            if requests>=max_requests:raise ValueError('Dynamic request budget exhausted')
            requests+=1;version=states[rid]['version'];context=dict(execution_history=str(states[rid]['steps']));invoked=time.monotonic_ns()
            value=await supervisor.predict(rid,mid,deepcopy(states[rid]['observation']),context,**kw)
            if value.get('request_id'):
                row=dict(request_id=value['request_id'],robot_id=rid,component_id=mid,status=value['status'],invoked_ns=invoked,returned_ns=time.monotonic_ns(),observation_directory='observations/'+str(list(states).index(rid))+'/'+str(version),context=context)
                request_log(row);observed.append(row)
            value['invoked_ns']=invoked
            return value
        async def execute(rid,action,duration,ticket_id):
            nonlocal capture_bytes
            begin=time.monotonic_ns();s=states[rid]
            if s['done']:return 0
            import numpy as np
            if capture_bytes+sum(v.nbytes for v in action.values())>max_capture_bytes:raise ValueError('Dynamic capture budget exhausted')
            target=out/('action-'+str(time.monotonic_ns())+'.npz');np.savez_compressed(target,**action)
            capture_bytes+=target.stat().st_size
            action_log(dict(kind='started',ticket_id=ticket_id,robot_id=rid,time_ns=time.monotonic_ns(),action_file=target.name,action_sha256=sha256(target),input_sha256=digest_value(action),before_version=s['version']))
            result=await call(simulators[rid].step(action,min(8,min(v.shape[1] for v in action.values()))))
            if type(result['steps']) is not int or not 0<result['steps']<=8:raise ValueError('Invalid simulator executed steps')
            if type(result['done']) is not bool or type(result['success']) is not bool:raise ValueError('Invalid simulator terminal flags')
            s.update(observation=result['observation'],version=s['version']+1,steps=s['steps']+result['steps'],done=result['done'],success=result['success']);capture(rid)
            action_log(dict(kind='stepped',ticket_id=ticket_id,robot_id=rid,time_ns=time.monotonic_ns(),executed_steps=result['steps'],after_version=s['version'],done=s['done'],success=s['success']))
            await asyncio.sleep(max(0,duration-(time.monotonic_ns()-begin)/1e9));return result['steps']
        async def goal(rid,c):
            model={roles[m['id']]:m['id'] for m in c['components']}
            while not stop.is_set() and time.monotonic_ns()<end:
                if states[rid]['done']:
                    # Terminal task outcome stays explicit; do not infer task success from throughput.
                    await sleep_until(end);return
                response=await request(rid,model['s1'])
                if response['status']=='needs_plan':await request(rid,model['s2'])
                elif response['status']=='ok' and not stop.is_set() and time.monotonic_ns()<end and (evaluation is None or response['invoked_ns']>=start):
                    admit()
                    ticket=response['ticket_id']
                    applied=await supervisor.apply(ticket,lambda rid,action,duration:execute(rid,action,duration,ticket))
                    if applied['status']=='unknown':raise RuntimeError('Physical action completion unknown')
                pause=.1 if response['status'] in ('blocked','not_active') else max(.01,min(1.,response.get('retry_after_s',.01)))
                await asyncio.sleep(pause)
        async def periodic(rid,m):
            period=round(1e9/m['periodic_hz']);index=max(0,(time.monotonic_ns()-epoch+period-1)//period)
            while not stop.is_set():
                tick=epoch+index*period
                if tick>=end:return
                await sleep_until(tick)
                if stop.is_set() or time.monotonic_ns()>=end:return
                index=max(index,(time.monotonic_ns()-epoch)//period);tick=epoch+index*period
                await request(rid,m['id'],scheduled_ns=tick,slot_index=index);index+=1
        async def transitions():
            for index,(offset,callback) in enumerate(sorted(changes,key=lambda v:v[0])):
                await sleep_until(start+round(offset*1e9))
                if stop.is_set():return
                admit();supervisor.emit('scheduled_change_started',index=index,offset_s=offset)
                value=await callback();supervisor.emit('scheduled_change_result',index=index,offset_s=offset,result=value)
        async def warmup_guard():
            await sleep_until(start)
            if not stop.is_set():admit()
        known={c['id']:c for c in supervisor.spec['catalog']['fleet']['classes']}
        for rid in simulators:
            c=known[class_by_robot[rid]];jobs.append(asyncio.create_task(goal(rid,c)))
            for m in c['components']:
                if roles[m['id']] in ('safety','monitor'):jobs.append(asyncio.create_task(periodic(rid,m)))
        jobs.append(asyncio.create_task(transitions()))
        if evaluation is not None:jobs.append(asyncio.create_task(warmup_guard()))
        # Surface worker-loop exceptions immediately, but retain the fixed cutoff.
        for job in jobs:
            def done(t):
                if not t.cancelled() and t.exception():errors.append(str(t.exception()) if str(t.exception())=='DynamicWarmupShortfall' else type(t.exception()).__name__);stop.set()
            job.add_done_callback(done)
        await sleep_until(end);stop.set()
        await asyncio.wait_for(asyncio.gather(*jobs,return_exceptions=True),supervisor.spec['deployment_timeout_s']+supervisor.spec['call_timeout_s']+max(c['action_time_s'] for c in known.values()))
        if errors:supervisor.blocked=True
        write_json(out/'result.json',dict(status='warmup_shortfall' if 'DynamicWarmupShortfall' in errors else 'failed' if errors else 'complete',errors=errors,request_attempts=requests,window_start_ns=start,window_end_ns=end,
            robot_outcomes={r:{k:v for k,v in s.items() if k!='observation'} for r,s in states.items()},task_success_is_simulator_reported_only=True))
    except BaseException:
        supervisor.blocked=True
        raise
    finally:
        stop.set()
        for task in jobs:
            if not task.done():task.cancel()
        await asyncio.gather(*jobs,return_exceptions=True)
        for sim in simulators.values():
            try:await sim.close()
            except Exception as exc:
                supervisor.blocked=True;errors.append('simulator_cleanup:'+type(exc).__name__);sim.abort()
        request_log.close();action_log.close()
        write_json(out/'cleanup.json',dict(status='failed' if any(e.startswith('simulator_cleanup:') for e in errors) else 'complete',errors=errors))
