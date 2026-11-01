"""Recompute retreat bounds, authorization, fresh inputs and separate metrics."""
import numpy as np
from .paired import digest_value
from .regression import sha256
from .profile_adoption import inside
from .recovery_policy import validate_config,valid_authorization,candidates,Limits
from .hybrid_safety import valid_receipt
from .vlm_contract import parse_generation


def audit_recovery(out,plan,records,intents,details,start,end):
    policy=validate_config(plan['recovery_policy']);guard=plan['collision_guard']
    def index(kind,key='request_id'):
        rows=[x for x in intents if x['kind']==kind];d={x[key]:x for x in rows}
        if len(d)!=len(rows):raise ValueError('Duplicate recovery evidence')
        return d
    attempts=index('recovery_decision','recovery_id');starts=index('recovery_started','recovery_id')
    launches=index('recovery_action_started','recovery_id');resumptions=index('recovery_s1_resumed','recovery_id')
    previews=index('collision_guard_preview');tensors=index('action_tensor');gates=index('hybrid_safety_gate')
    actions={x['request_id']:x for x in records if x['record_type']=='action'}
    requests={x['request_id']:x for x in records if x['record_type']=='request'}
    robot_classes={r['robot_id']:c['audit'] for c in plan['classes'] for r in c['robots']}
    used=set();retreats=[];by_episode={};parent_ids=set()
    def load(row):
        path=inside(out,row['file'])
        if sha256(path)!=row['sha256']:raise ValueError('Recovery tensor hash differs')
        with np.load(path,allow_pickle=False) as f:return dict(f)
    def visual(rid,eid,at,reference,minimum,exact=False):
        comp=next(c for c in robot_classes[rid]['components'] if c['role']=='safety')
        rows=[r for r in requests.values() if (r['robot_id'],r['episode_id'],r['component_id'])==(rid,eid,comp['id']) and r['finished_ns']<=at]
        r=max(rows,key=lambda x:x['finished_ns']) if rows else None
        if r is None or r['request_id']!=reference or r['status']!='ok' or not r['valid_output']:raise ValueError('Latest recovery visual result missing')
        d=details[reference];b=d.get('backend_evidence') or {}
        parsed=parse_generation('safety',b.get('raw_generation',''),finish_reason=b.get('finish_reason'),format='safety_visual_guard_v1')
        deadline=r['scheduled_ns']+round(comp['deadline_ms']*1e6)
        if parsed['stop'] or parsed!=d['semantic_output'] or digest_value(parsed)!=d['output_sha256'] or r['finished_ns']>deadline or at>=deadline+comp['period_ns']:
            raise ValueError('Invalid/expired recovery visual authorization')
        if d['observation_version']<minimum or (exact and d['observation_version']!=minimum):raise ValueError('Stale recovery visual observation')
    for a in actions.values():
        if not a['applied'] or a['request_id'] not in previews:continue
        rid=a['request_id'];tensor=load(tensors[rid]);proof=previews[rid]['evidence']
        if not valid_authorization(proof,guard,policy,tensor,tensors[rid]['requested_steps'],'s1'):raise ValueError('S1 terminal retreat witness invalid')
        if proof['lease_id'] in used:raise ValueError('Reused motion lease')
        used.add(proof['lease_id'])
    if set(starts)!=set(attempts):raise ValueError('Incomplete recovery attempts')
    for rid,e in attempts.items():
        parent=e['parent_request_id'];a=actions.get(parent);begin=starts[rid]
        if parent in parent_ids or a is None or a['applied'] or not a['safety_stopped']:raise ValueError('Retreat must follow one rejected S1 action')
        parent_ids.add(parent)
        if any(e[k]!=a[k] for k in ('robot_id','episode_id')) or e['motion_source']!='retreat' or e['counts_as_S1_success'] is not False:
            raise ValueError('Recovery identity/source differs')
        if any(begin[k]!=e[k] for k in ('parent_request_id','robot_id','episode_id','start_ns','observation_version','requested_steps')):
            raise ValueError('Recovery intent differs')
        if not a['end_ns']<=e['start_ns']<=e['preview_end_ns']<=e['end_ns'] or previews[parent]['evidence']['status']!='blocked':
            raise ValueError('Unknown or out-of-order motion recovered')
        if not e['applied']:
            if e['executed_steps']!=0 or rid in launches:raise ValueError('Unaccounted recovery motion')
            continue
        proof=e['response']['evidence'];template=load(tensors[parent]);action=load(e)
        choices=dict(candidates(template,policy));name=e['response'].get('candidate')
        if name not in choices or digest_value(action)!=digest_value(choices[name]) or digest_value(action)!=e['action_sha256']:
            raise ValueError('Retreat candidate differs')
        if not valid_authorization(proof,guard,policy,action,e['requested_steps'],'retreat') or not valid_receipt(e['receipt'],proof,e['executed_steps']):
            raise ValueError('Retreat geometry proof/receipt invalid')
        if proof['lease_id'] in used:raise ValueError('Reused retreat lease')
        used.add(proof['lease_id'])
        if rid not in launches or launches[rid]['start_ns']!=e['action_start_ns'] or not e['preview_end_ns']<=e['action_start_ns']<end or e['new_observation_version']!=e['observation_version']+1:
            raise ValueError('Recovery motion/observation ordering differs')
        visual(e['robot_id'],e['episode_id'],e['action_start_ns'],e['visual_request_id'],e['observation_version'],exact=True)
        retreats.append(e)
    if set(launches)!={e['recovery_id'] for e in retreats}:raise ValueError('Unresolved retreat launch')
    timeline=[dict(x,source='s1') for x in actions.values() if x['applied']]+[dict(x,source='retreat') for x in retreats]
    expected_resumptions={}
    for e in sorted(timeline,key=lambda x:x.get('action_start_ns',x['start_ns'])):
        key=(e['robot_id'],e['episode_id']);state=by_episode.setdefault(key,dict(limits=Limits(policy),last=None,streak=0))
        limits=state['limits']
        if e['source']=='retreat':
            proof=e['response']['evidence'];delta=proof.get('end_effector_delta_m',[])
            if not limits.available() or not limits.accepts(delta):raise ValueError('Recovery motion exceeds limits')
            distance=float(np.linalg.norm(delta))
            if abs(distance-proof['displacement_m'])>1e-12:raise ValueError('Recovery distance differs')
            limits.applied('retreat',distance)
            expected=dict(policy_sha256=digest_value(policy),motion_source='retreat',**limits.evidence())
            if e['recovery_state']!=expected:raise ValueError('Recovery counters differ')
            state.update(last=e,streak=0)
        else:
            limits.applied('s1');last=state['last']
            if last:
                d=details[e['request_id']];minimum=last['new_observation_version']
                if d['observation_version']<minimum:raise ValueError('Stale S1 observation after retreat')
                visual(e['robot_id'],e['episode_id'],e['start_ns'],gates[e['request_id']]['visual_request_id'],minimum)
                if robot_classes[e['robot_id']]['id']=='P4':
                    pid=d['plan_request_id']
                    if pid not in requests or requests[pid]['sent_ns']<=last['action_start_ns'] or details[pid]['observation_version']<minimum:
                        raise ValueError('P4 reused pre-retreat plan')
                state['streak']+=1
                if state['streak']==policy['resumed_s1_chunks_required']:expected_resumptions[last['recovery_id']]=e['request_id']
    if {k:v['request_id'] for k,v in resumptions.items()}!=expected_resumptions:raise ValueError('S1 resumption evidence differs')
    result={}
    for cid in ('P3','P4'):
        rids={rid for rid,c in robot_classes.items() if c['id']==cid}
        original=[a for a in actions.values() if a['robot_id'] in rids and start<=a['start_ns']<end
            and not requests[a['request_id']]['warmup'] and start<=requests[a['request_id']]['sent_ns']<end]
        rejected=[a for a in original if a['safety_stopped']];ids={a['request_id'] for a in rejected}
        applied=[e for e in retreats if e['parent_request_id'] in ids]
        n=len(original)
        result[cid]=dict(original_s1_decisions=n,original_s1_refusals=len(rejected),original_s1_refusal_rate=len(rejected)/n if n else None,
            original_s1_applied=sum(a['applied'] for a in original),retreat_attempts=sum(e['parent_request_id'] in ids for e in attempts.values()),
            retreats_applied=len(applied),final_stops=len(rejected)-len(applied),final_stop_rate=(len(rejected)-len(applied))/n if n else None,
            sustained_s1_resumptions=sum(e['recovery_id'] in expected_resumptions for e in applied),
            censored_or_unrecovered_retreats=sum(e['recovery_id'] not in expected_resumptions for e in applied))
    return dict(classes=result,counts_retreat_as_s1_success=False,simulation_only=True,
        resumption_definition='three subsequent applied S1 chunks before another retreat; cutoff is not success',
        final_stop_definition='rejected original S1 decision without an immediately applied verified retreat'),retreats
