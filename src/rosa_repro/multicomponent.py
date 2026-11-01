"""P1-P4 event audit. The slot/action linkage is a declared project convention,
not a claim of equivalence to unpublished paper evaluation code.
"""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import math

from .contracts import (obj,TEXT,INTEGER,POSINT,NUMBER,POSITIVE,REQUEST_SCHEMA,
                        ACTION_SCHEMA,validate)

REQ2=deepcopy(REQUEST_SCHEMA)
REQ2['properties'].update(schema_version={'const':'2.0'},
    slot_index={'type':['integer','null'],'minimum':0},plan_request_id={'type':['string','null'],'minLength':1})
REQ2['required']+=['slot_index','plan_request_id']
ACT2=deepcopy(ACTION_SCHEMA)
ACT2['properties'].update(schema_version={'const':'2.0'},safety_stopped={'type':'boolean'})
ACT2['required']+=['safety_stopped']
EVENT_SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema','oneOf':[REQ2,ACT2]}
COMPONENT=obj({'id':TEXT,'role':{'enum':['s1','s2','safety','monitor']},'deadline_ms':POSITIVE,
               'period_ns':POSINT,'phase_ns':INTEGER},['id','role','deadline_ms'])
CLASS=obj({'id':TEXT,'pipeline':{'enum':['P1','P2','P3','P4']},'weight':NUMBER,
           's2_horizon':POSINT,'components':{'type':'array','minItems':1,'items':COMPONENT}})
EPISODE=obj({'class_id':TEXT,'robot_id':TEXT,'episode_id':TEXT,'start_ns':INTEGER,'end_ns':INTEGER})
INPUT_SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema',**obj({
    'schema_version':{'const':'2.0'},'run_id':TEXT,'clock_domain':TEXT,'run_epoch_ns':INTEGER,
    'window_start_ns':INTEGER,'window_end_ns':INTEGER,'audit_end_ns':INTEGER,
    'classes':{'type':'array','minItems':1,'items':CLASS},
    'episodes':{'type':'array','items':EPISODE},'records':{'type':'array','items':EVENT_SCHEMA}})}
ROLES={'P1':{'s1'},'P2':{'s1','monitor'},'P3':{'s1','safety','monitor'},'P4':{'s1','s2','safety','monitor'}}


def ns_from_ms(value):
    ns=Decimal(str(value))*1_000_000
    if ns!=int(ns):raise ValueError('Deadline cannot be represented in integer nanoseconds')
    return int(ns)


def paper_class(cid,pipeline,weight=1):
    c={'id':cid,'pipeline':pipeline,'weight':weight,'s2_horizon':10,'components':[]}
    deadlines={'s1':200,'s2':2000,'safety':500,'monitor':2000}
    for role in ('s1','s2','safety','monitor'):
        if role not in ROLES[pipeline]:continue
        spec={'id':role,'role':role,'deadline_ms':deadlines[role]}
        if role in ('safety','monitor'):
            spec.update(period_ns=500_000_000 if role=='safety' else 2_000_000_000,phase_ns=0)
        c['components'].append(spec)
    return c


def evaluate_multicomponent(data):
    validate(data,INPUT_SCHEMA)
    start,end,audit=data['window_start_ns'],data['window_end_ns'],data['audit_end_ns']
    if not start<end<=audit:raise ValueError('Invalid explicit measurement/audit window')
    classes={c['id']:c for c in data['classes']}
    if len(classes)!=len(data['classes']):raise ValueError('Duplicate class ID')
    for cls in classes.values():
        comps=cls['components']
        if len({c['id'] for c in comps})!=len(comps) or {c['role'] for c in comps}!=ROLES[cls['pipeline']] or len(comps)!=len(ROLES[cls['pipeline']]):
            raise ValueError('Pipeline components/roles must match P1-P4 exactly')
        for c in comps:
            ns_from_ms(c['deadline_ms'])
            periodic=c['role'] in ('safety','monitor')
            if periodic != ('period_ns' in c and 'phase_ns' in c):raise ValueError('Periodic fields required only for obligations')
            if not periodic and ('period_ns' in c or 'phase_ns' in c):raise ValueError('Nonperiodic component has period fields')
            if periodic and c['phase_ns']>=c['period_ns']:raise ValueError('Phase must be below period')
    episodes={};by_robot={}
    for e in data['episodes']:
        key=(e['robot_id'],e['episode_id'])
        if key in episodes or e['class_id'] not in classes:raise ValueError('Duplicate episode or unknown class')
        if not data['run_epoch_ns']<=e['start_ns']<e['end_ns']:raise ValueError('Invalid episode time range')
        episodes[key]=e;by_robot.setdefault(e['robot_id'],[]).append(e)
    for es in by_robot.values():
        es.sort(key=lambda e:e['start_ns'])
        if any(a['end_ns']>b['start_ns'] for a,b in zip(es,es[1:])):raise ValueError('Overlapping robot episodes')
    def context(r):
        if r['run_id']!=data['run_id'] or r['clock_domain']!=data['clock_domain']:raise ValueError('Mixed run/clock')
        key=(r['robot_id'],r['episode_id']);e=episodes.get(key)
        if e is None:raise ValueError('Event has no declared episode')
        cls=classes[e['class_id']]
        c=next((c for c in cls['components'] if c['id']==r['component_id']),None)
        if c is None:raise ValueError('Component is not required by this class')
        return key,e,cls,c
    requests={};actions=[];periodic_attempts={};s1_groups={}
    for r in data['records']:
        key,e,cls,c=context(r)
        if r['record_type']=='action':
            if c['role']!='s1':raise ValueError('Only S1 creates action chunks')
            actions.append(r);continue
        if r['request_id'] in requests:raise ValueError('Duplicate request ID')
        if not e['start_ns']<=r['scheduled_ns']<=r['sent_ns']<e['end_ns'] or not r['sent_ns']<=r['finished_ns']<=audit:
            raise ValueError('Invalid request time or unfinalized audit')
        if r['deadline_ms']!=c['deadline_ms']:raise ValueError('Request deadline disagrees with declaration')
        if r['status']!='ok' and r['valid_output']:raise ValueError('Failed request marked valid')
        if (r['batch_id'] is None)!=(r['actual_batch_size'] is None):raise ValueError('Incomplete batch metadata')
        if c['role'] in ('safety','monitor'):
            if r['slot_index'] is None or r['plan_request_id'] is not None:raise ValueError('Invalid periodic dependency fields')
            expected=data['run_epoch_ns']+c['phase_ns']+r['slot_index']*c['period_ns']
            if r['scheduled_ns']!=expected:raise ValueError('Request assigned to wrong scheduled slot')
            periodic_attempts.setdefault((*key,c['id'],r['slot_index']),[]).append(r)
        else:
            if r['slot_index'] is not None:raise ValueError('Nonperiodic request assigned to a slot')
            if c['role']=='s2' and r['plan_request_id'] is not None:raise ValueError('S2 cannot depend on itself')
            if c['role']=='s1':
                if cls['pipeline']!='P4' and r['plan_request_id'] is not None:raise ValueError('Unexpected S2 dependency')
                s1_groups.setdefault(key,[]).append(r)
        requests[r['request_id']]=r
    def valid(r):return r['status']=='ok' and r['valid_output']
    def ontime(r):return valid(r) and r['finished_ns']-r['sent_ns']<=ns_from_ms(r['deadline_ms'])
    # Every S1 invocation consumes one plan use, including warmup/failure/retry.
    plan_use={};seen_uses=Counter()
    for key,rs in s1_groups.items():
        rs.sort(key=lambda r:(r['sent_ns'],r['request_id']))
        if any(a['sent_ns']==b['sent_ns'] or a['finished_ns']>b['sent_ns'] for a,b in zip(rs,rs[1:])):
            raise ValueError('Synchronous S1 ordering is ambiguous/overlapping')
        for r in rs:
            pid=r['plan_request_id']
            if pid is None:continue
            p=requests.get(pid)
            if p is not None:
                _,pe,pc,component=context(p)
                if (p['robot_id'],p['episode_id'])!=key or component['role']!='s2':raise ValueError('Cross-episode/robot or non-S2 plan reference')
            seen_uses[(*key,pid)]+=1;plan_use[r['request_id']]=seen_uses[(*key,pid)]
    ledgers={};slots=[];startup={};slot_count=0
    for key,e in episodes.items():
        cls=classes[e['class_id']];ready=e['start_ns']
        for c in cls['components']:
            if c['role'] not in ('safety','monitor'):continue
            base=data['run_epoch_ns']+c['phase_ns'];period=c['period_ns'];d=ns_from_ms(c['deadline_ms'])
            first=max(0,(e['start_ns']-base+period-1)//period)
            # Generate only slots whose planned send occurs while episode exists.
            limit=min(e['end_ns'],audit+1)
            last=(limit-1-base)//period
            count=max(0,last-first+1);slot_count+=count
            if slot_count>100000:raise ValueError('Audit exceeds 100000-slot CPU limit')
            ready=max(ready,base+first*period+d)
            ledger=[]
            for i in range(first,last+1):
                scheduled=base+i*period;deadline=scheduled+d
                attempts=periodic_attempts.get((*key,c['id'],i),[])
                winners=[r for r in attempts if valid(r) and r['finished_ns']<=deadline]
                state=('pending' if deadline>audit else 'qualified' if winners else 'missing' if not attempts else 'violated')
                row=dict(robot_id=key[0],episode_id=key[1],class_id=cls['id'],component_id=c['id'],slot_index=i,
                    scheduled_ns=scheduled,deadline_ns=deadline,status=state,attempts=len(attempts),
                    qualifying_request_ids=[r['request_id'] for r in winners],
                    scheduled_to_finished_ms=[(r['finished_ns']-scheduled)/1e6 for r in attempts])
                ledger.append(row);slots.append(row)
            ledgers[(*key,c['id'])]=([s['deadline_ns'] for s in ledger],ledger)
        startup[key]=max(0,min(ready,e['end_ns'],end)-max(e['start_ns'],start))/1e9
    per_action=[];seen_actions=set();seen_requests=set();applied_by_robot={}
    totals={cid:Counter() for cid in classes}
    for a in sorted(actions,key=lambda a:(a['start_ns'],a['action_id'])):
        key,e,cls,c=context(a)
        if a['action_id'] in seen_actions or a['request_id'] in seen_requests:raise ValueError('Repeated action application')
        seen_actions.add(a['action_id']);seen_requests.add(a['request_id'])
        r=requests.get(a['request_id'])
        if r is None or any(a[k]!=r[k] for k in ('robot_id','episode_id','component_id')):raise ValueError('Action/request identity mismatch')
        if not r['finished_ns']<=a['start_ns']<=a['end_ns']<=audit:raise ValueError('Action timestamps invalid')
        if not e['start_ns']<=a['start_ns']<e['end_ns'] or a['end_ns']>e['end_ns']:raise ValueError('Action survived episode reset')
        if a['applied']!=(a['executed_steps']>0):raise ValueError('Action steps/applied disagree')
        if a['applied'] and (not valid(r) or a['safety_stopped']):raise ValueError('Invalid/stopped action was applied')
        if a['applied']:
            prev=applied_by_robot.get(a['robot_id'])
            if prev and prev['end_ns']>a['start_ns']:raise ValueError('Robot actions overlap')
            # In the synchronous contract inference cannot overlap prior action.
            if prev and prev['end_ns']>r['sent_ns']:raise ValueError('Inference overlaps prior action')
            applied_by_robot[a['robot_id']]=a
        reasons=[]
        if not ontime(r):reasons.append('s1_deadline_or_invalid')
        if cls['pipeline']=='P4':
            p=requests.get(r['plan_request_id'])
            if p is None:reasons.append('s2_plan_missing')
            else:
                if p['finished_ns']>r['sent_ns']:reasons.append('s2_plan_not_ready')
                if not ontime(p):reasons.append('s2_deadline_or_invalid')
                if plan_use[r['request_id']]>cls['s2_horizon']:reasons.append('s2_horizon_exceeded')
        linked={}
        for periodic in cls['components']:
            if periodic['role'] not in ('safety','monitor'):continue
            deadlines,ledger=ledgers[(*key,periodic['id'])]
            i=bisect_right(deadlines,a['start_ns'])-1
            if i<0:
                reasons.append(periodic['role']+'_startup_pending');linked[periodic['id']]=None
            else:
                slot=ledger[i];linked[periodic['id']]=slot['slot_index']
                if slot['status']!='qualified':reasons.append(periodic['role']+'_slot_'+slot['status'])
        measured=(not r['warmup'] and start<=r['sent_ns']<end and start<=a['start_ns']<end)
        t=totals[cls['id']]
        if measured:
            t['decisions']+=1;t['safety_stopped']+=a['safety_stopped']
            if a['applied']:
                t['raw']+=1;t['s1_qualified']+=ontime(r);t['qualified']+=not reasons
                for reason in reasons:t['reason:'+reason]+=1
        elif a['applied'] and start<=a['start_ns']<end:t['cross_boundary_or_warmup_excluded']+=1
        per_action.append(dict(action_id=a['action_id'],class_id=cls['id'],request_id=r['request_id'],
            measured=measured,applied=a['applied'],safety_stopped=a['safety_stopped'],
            model_qualified=not reasons,qualified_and_counted=measured and a['applied'] and not reasons,
            reasons=reasons,periodic_slot_links=linked,plan_use_index=plan_use.get(r['request_id'])))
    duration=(end-start)/1e9
    class_results={}
    for cid,cls in classes.items():
        t=totals[cid];class_results[cid]=dict(pipeline=cls['pipeline'],weight=cls['weight'],
            raw_action_chunks=t['raw'],s1_qualified_action_chunks=t['s1_qualified'],qualified_action_chunks=t['qualified'],
            qualified_action_rate_hz=t['qualified']/duration,weighted_Q=cls['weight']*t['qualified']/duration,
            safety_stopped_decisions=t['safety_stopped'],decisions=t['decisions'],
            excluded_cross_boundary_or_warmup=t['cross_boundary_or_warmup_excluded'],
            startup_robot_seconds=sum(v for k,v in startup.items() if episodes[k]['class_id']==cid),
            disqualification_reasons={k[7:]:v for k,v in t.items() if k.startswith('reason:')})
    request_metrics={};slot_metrics={}
    for cid,cls in classes.items():
        request_metrics[cid]={};slot_metrics[cid]={}
        for c in cls['components']:
            cohort=[r for r in requests.values() if episodes[(r['robot_id'],r['episode_id'])]['class_id']==cid
                    and r['component_id']==c['id'] and not r['warmup'] and start<=r['sent_ns']<end]
            lat=sorted((r['finished_ns']-r['sent_ns'])/1e6 for r in cohort if valid(r))
            p99=lat[math.ceil(.99*len(lat))-1] if lat else None
            request_metrics[cid][c['id']]=dict(requests=len(cohort),status_counts=dict(Counter(r['status'] for r in cohort)),
                send_deadline_met=sum(ontime(r) for r in cohort),send_p99_ms_valid_only=p99,
                send_p99_within_deadline=p99<=c['deadline_ms'] if p99 is not None else None,p99_exploratory=len(lat)<1000)
            matured=[s for s in slots if s['class_id']==cid and s['component_id']==c['id'] and start<=s['deadline_ns']<end]
            if c['role'] in ('safety','monitor'):
                slot_metrics[cid][c['id']]=dict(expected_matured_slots=len(matured),status_counts=dict(Counter(s['status'] for s in matured)),
                    slot_meet_rate=sum(s['status']=='qualified' for s in matured)/len(matured) if matured else None)
    return dict(schema_version='2.0',metric_protocol='declared-slot-and-plan-p1-p4-v1',run_id=data['run_id'],
        duration_s=duration,weighted_Q=sum(c['weighted_Q'] for c in class_results.values()),
        qualified_action_chunks=sum(c['qualified_action_chunks'] for c in class_results.values()),
        classes=class_results,component_request_metrics=request_metrics,periodic_slot_metrics=slot_metrics,
        actions=per_action,slots=slots,paper_official_metric_equivalence_verified=False,
        gpu_execution=False,evaluation_input_provenance='caller-supplied events; evaluator performs no model inference')
