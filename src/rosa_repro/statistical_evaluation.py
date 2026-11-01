"""Auditable uncertainty for paired tasks and dependent request latency.
Intervals are conditional on stated sampling assumptions, never automatic admission.
"""
from collections import Counter
import json,math
from pathlib import Path
import numpy as np
from .execution_plan import digest
from .regression import sha256
from .profile_adoption import inside


def exact_binomial_interval(k,n,confidence=.95):
    """Clopper-Pearson via monotone binomial-CDF inversion, including boundary counts."""
    if type(n) is not int or not 1<=n<=10000 or type(k) is not int or not 0<=k<=n or not .5<confidence<1:
        raise ValueError('Invalid binomial interval controls')
    logchoose=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1) for j in range(n+1)]
    def cdf(j,p):
        if p==0:return 1.
        if p==1:return float(j==n)
        logs=[logchoose[i]+i*math.log(p)+(n-i)*math.log1p(-p) for i in range(j+1)]
        top=max(logs);return min(1.,math.exp(top)*sum(math.exp(v-top) for v in logs))
    def inverse(j,target):
        low,high=0.,1.
        for _ in range(64):
            mid=(low+high)/2
            if cdf(j,mid)>target:low=mid
            else:high=mid
        return (low+high)/2
    tail=(1-confidence)/2
    return [0. if k==0 else inverse(k-1,1-tail),1. if k==n else inverse(k,tail)]


def paired_success(reference,candidate,*,confidence=.95,margin=.05,trial_complete=True):
    if not .5<confidence<1 or not 0<margin<1 or not reference or len(reference)!=len(candidate):raise ValueError('Invalid paired comparison')
    key=lambda o:(o['task'],o['state_sha256'],o['seed'])
    refs={key(o):o for o in reference};cands={key(o):o for o in candidate}
    if len(refs)!=len(reference) or len(cands)!=len(candidate) or refs.keys()!=cands.keys():raise ValueError('Initial-state/seed sets differ or repeat')
    n=len(refs);pairs=[];mismatches=[];delta_lo=delta_hi=0
    for k,a in refs.items():
        b=cands[k]
        for o in (a,b):
            if (o['status']=='completed' and type(o['success']) is not bool) or (o['status']!='completed' and o['success'] is not None):
                raise ValueError('Invalid success evidence')
        if a.get('initial_observation_sha256') and b.get('initial_observation_sha256'):
            if any(a.get(f)!=b.get(f) for f in ('initial_observation_sha256','restored_state_sha256')):mismatches.append(list(k))
        av=int(a['success']) if a['status']=='completed' else None
        bv=int(b['success']) if b['status']=='completed' else None
        delta_lo+=(bv if bv is not None else 0)-(av if av is not None else 1)
        delta_hi+=(bv if bv is not None else 1)-(av if av is not None else 0)
        if av is not None and bv is not None:pairs.append((av,bv))
    complete=trial_complete and len(pairs)==n and not mismatches and all(
        a.get('initial_observation_sha256') and a.get('restored_state_sha256') and b.get('initial_observation_sha256') and b.get('restored_state_sha256')
        for a,b in ((refs[k],cands[k]) for k in refs))
    counts=Counter(pairs);gains=counts[(0,1)];losses=counts[(1,0)]
    ci=None;point=None;decision='unresolved'
    if complete:
        # Union bound: each discordance proportion has coverage 1-alpha/2.
        # Their dependence is retained in the paired multinomial counts; no independence between proportions is assumed.
        g=exact_binomial_interval(gains,n,(1+confidence)/2);l=exact_binomial_interval(losses,n,(1+confidence)/2)
        ci=[max(-1.,g[0]-l[1]),min(1.,g[1]-l[0])];point=(gains-losses)/n
        decision='investigate_drop' if point < -margin else 'within_margin_under_sampling_assumption' if ci[0]>-margin and ci[1]<margin else 'unresolved'
    return dict(planned_pairs=n,complete_pairs=len(pairs),all_pairs_valid=bool(complete),initial_condition_mismatches=mismatches,
        reference_successful_all_planned=sum(o['success'] is True for o in reference),candidate_successful_all_planned=sum(o['success'] is True for o in candidate),
        discordant_gain=gains,discordant_loss=losses,success_difference=point,confidence=confidence,confidence_interval=ci,
        missing_outcome_identification_bounds=None if mismatches else [delta_lo/n,delta_hi/n],margin=margin,decision=decision,
        method='paired_discordance_Clopper_Pearson_with_Bonferroni_union_bound',sampling_unit='one matched task/initial-state/seed pair',
        interval_scope='per comparison; not simultaneous across tasks/widths/arms',random_seed=None,
        limitations=['Conservative interval assumes independent representative case pairs; a fixed benchmark bank does not establish population representativeness.',
          'One model seed per state does not quantify repeated policy-noise variability. Batch-shape RNG alignment is unverified.',
          'Missing or failed trials suppress inference; all planned cases remain in denominators.'])


def r06_report(suite,root,directories,*,confidence=.95,margin=.05):
    from .r06_runtime import reaudit
    if set(directories)!={c['id'] for c in suite['cells']} or len(set(directories.values()))!=len(directories):raise ValueError('Prospective directory mapping must cover each cell once')
    rows=[];results={};sources=set();runids=set()
    for cell in suite['cells']:
        cid=cell['id'];row=dict(cell=cell,directory=directories[cid]);path=inside(root,directories[cid])
        if not path.exists():row.update(status='missing',reason='expected_directory_absent')
        else:
            try:
                audit=reaudit(path);r=json.loads((path/'result.json').read_text());source=json.loads((path/'source_manifest.json').read_text())
                if source!=suite['source_manifest']:raise ValueError('Source differs from suite')
                if r['suite_sha256']!=suite['sha256'] or r['cell']!=cell or json.loads((path/'suite.json').read_text())!=suite:raise ValueError('R06 suite/cell differs')
                if r['run_id'] in runids:raise ValueError('Run reused across cells')
                runids.add(r['run_id']);sources.add(digest(source));results[cid]=r
                row.update(status=r['status'],audit=audit,planned_cases=r['planned_cases'],completed_cases=r['completed_cases'],successful_cases=r['successful_cases'])
            except (ValueError,KeyError,TypeError,OSError) as exc:row.update(status='invalid',reason=str(exc))
        rows.append(row)
    comparisons=[]
    for cell in suite['cells']:
        if cell['arm']=='official':continue
        cid=cell['id'];rid=f"{cell['task']}-{cell['concurrency']}-official";a=results.get(rid);b=results.get(cid)
        row=dict(reference=rid,candidate=cid,task=cell['task'],concurrency=cell['concurrency'])
        if a and b and len(sources)==1:
            row['statistics']=paired_success(a['outcomes'],b['outcomes'],confidence=confidence,margin=margin,
                trial_complete=a['status']==b['status']=='complete')
        else:row.update(statistics=None,reason='missing_invalid_trial_or_mixed_sources')
        comparisons.append(row)
    return dict(schema_version='1.0',suite_sha256=suite['sha256'],directory_mapping_sha256=digest(directories),
        mode=suite['mode'],rows=rows,comparisons=comparisons,status='complete' if all(r['status']=='complete' for r in rows) else 'incomplete',
        automatic_admission=False,LIBERO_results=suite['mode']=='libero_gpu',pooling_across_widths_or_tasks=False)


def latency_blocks(rows,*,start_ns,end_ns,block_ns,seed=42,resamples=2000,confidence=.95,min_blocks=8):
    if any(type(v) is not int for v in (start_ns,end_ns,block_ns,seed,resamples,min_blocks)) or start_ns<0 or not end_ns>start_ns or block_ns<=0:
        raise ValueError('Integer monotonic window and positive block duration required')
    if not 0<=seed<2**32 or not 100<=resamples<=10000 or not .5<confidence<1 or min_blocks<8:
        raise ValueError('Seeded bounded resampling and at least eight blocks required')
    duration=end_ns-start_ns
    nblocks=max(1,duration//block_ns)
    # Use the entire fixed window. Equalized block lengths differ by at most one nanosecond.
    boundaries=[(i*duration+nblocks-1)//nblocks for i in range(nblocks+1)]
    if nblocks>100000:raise ValueError('Too many time blocks')
    ids=set();values=[];blocks=[];statuses=Counter();all_block_counts=[0]*nblocks
    for r in rows:
        if set(r)!={'request_id','cohort_ns','latency_ms','status'}:raise ValueError('Explicit normalized latency row required')
        if type(r['cohort_ns']) is not int or not start_ns<=r['cohort_ns']<end_ns:raise ValueError('Row outside declared measurement cohort')
        if not isinstance(r['request_id'],str) or not r['request_id'] or r['request_id'] in ids:raise ValueError('Duplicate/invalid request')
        ids.add(r['request_id']);status=r['status'];statuses[status]+=1
        if status not in ('ok','error','timeout','rejected','cancelled','not_sent','invalid'):raise ValueError('Unknown request status')
        block=(r['cohort_ns']-start_ns)*nblocks//duration;all_block_counts[block]+=1
        if status=='ok':
            v=r['latency_ms']
            if type(v) not in (float,int) or not math.isfinite(v) or v<0:raise ValueError('Invalid successful latency')
            values.append(v);blocks.append(block)
        elif r['latency_ms'] is not None:raise ValueError('Failure must not enter successful-only latency distribution')
    quantiles=(.5,.95,.99);point=lambda v,q:float(sorted(v)[max(0,math.ceil(len(v)*q)-1)]) if len(v) else None
    result=dict(method='nonoverlapping_equal_wall_time_block_percentile_bootstrap',sampling_unit='all requests/robots in one contiguous wall-time block',
        confidence=confidence,seed=seed,resamples=resamples,requested_block_ns=block_ns,start_ns=start_ns,end_ns=end_ns,
        actual_block_duration_ns_range=[min(b-a for a,b in zip(boundaries,boundaries[1:])),max(b-a for a,b in zip(boundaries,boundaries[1:]))],
        partition_rule='floor(window_duration/requested_block_duration) equalized whole-window blocks; no discarded tail',
        total_blocks=nblocks,blocks_with_success=len(set(blocks)),block_arrival_counts=all_block_counts,
        planned_logged_arrivals=len(rows),status_counts=dict(statuses),successful=len(values),
        metrics={f'p{int(q*100)}':dict(estimate_ms=point(values,q),confidence_interval_ms=None) for q in quantiles},
        status='unavailable',reason=None,p99_exploratory=len(values)<1000,failed_arrivals_excluded_only_from_latency=True,
        limitations=['Blocks must exceed relevant dependence timescales and the window must be approximately stationary; neither assumption is automatically certified.',
          'Intervals are within-run, conditional on successful responses. They do not certify whole-workload SLOs or between-run uncertainty.',
          'Block-length sensitivity must be reviewed; frames/robots/requests are not independent replicates.'])
    if nblocks<min_blocks or len(set(blocks))<min_blocks:result['reason']='insufficient_time_blocks_with_success';return result
    if (len(values)+nblocks)*resamples>50_000_000:raise ValueError('Resampling work budget exceeded; predeclare a bounded analysis')
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];block=np.asarray(blocks,dtype=int)[order]
    rng=np.random.default_rng(seed);samples=[[] for _ in quantiles];empty=0
    for _ in range(resamples):
        multiplicity=np.bincount(rng.integers(0,nblocks,size=nblocks),minlength=nblocks)
        cumulative=np.cumsum(multiplicity[block]);n=int(cumulative[-1])
        if not n:empty+=1;continue
        for k,q in enumerate(quantiles):samples[k].append(float(v[np.searchsorted(cumulative,math.ceil(q*n),side='left')]))
    result['empty_resamples']=empty
    if empty:result['reason']='resamples_without_success_no_conditional_dropping';return result
    for k,q in enumerate(quantiles):
        result['metrics'][f'p{int(q*100)}']['confidence_interval_ms']=[point(samples[k],(1-confidence)/2),point(samples[k],(1+confidence)/2)]
    result.update(status='estimated',reason=None);return result


def latency_report(directory,*,kind,block_ns,seed=42,resamples=2000,confidence=.95):
    p=Path(directory);read=lambda n:json.loads((p/n).read_text());groups={};extra={}
    if kind=='profile':
        from .profile_runner import reaudit,verify_evidence
        reaudit(p);seal=verify_evidence(p);result=read('result.json');window=result['measurement']
        raw=[json.loads(x) for x in (p/'requests.jsonl').read_text().splitlines()]
        cohort=[r for r in raw if r['phase']=='measure' and window['start_ns']<=r['scheduled_ns']<window['end_ns']]
        groups['component']=[dict(request_id=r['request_id'],cohort_ns=r['scheduled_ns'],status=r['status'],
            latency_ms=(r['finished_ns']-r['sent_ns'])/1e6 if r['status']=='ok' else None) for r in cohort]
        binding=dict(kind=kind,evidence_sha256=seal['sha256']);extra['cohort_definition']='scheduled measurement arrivals; includes not_sent'
    elif kind=='r05':
        from .r05_runtime import reaudit
        audit=reaudit(p);binding=dict(kind=kind,evidence_sha256=audit['evidence_sha256']);result=read('result.json')
        if not audit['measurement_started']:return dict(status='unavailable',reason='no_measurement_started',binding=binding,groups={})
        window=next(w for w in result['windows'] if w['phase']=='measurement')
        rows=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()]
        starts={r['request_id']:r for r in rows if r['kind']=='request_start' and r['phase']=='measurement'}
        ends={r['request_id']:r for r in rows if r['kind']=='request_end' and r['phase']=='measurement'}
        if starts.keys()!=ends.keys():return dict(status='unavailable',reason='unfinalized_requests',binding=binding,groups={})
        groups['s1']=[dict(request_id=qid,cohort_ns=q['time_ns'],status=ends[qid]['status'],
            latency_ms=(ends[qid]['time_ns']-q['time_ns'])/1e6 if ends[qid]['status']=='ok' else None) for qid,q in starts.items()]
        extra=dict(cohort_definition='R05 measurement sends, including post-cutoff completions; no latency-SLO qualification',trial_status=audit['status'],latency_slo_ms=None)
    elif kind=='pipeline':
        from .comparison_report import audit_pipeline
        audit=audit_pipeline(p);binding=dict(kind=kind,files={f.name:sha256(f) for f in sorted(p.iterdir()) if f.is_file()})
        if 'request_metrics' not in audit:return dict(status='unavailable',reason='no_completed_measurement',audit=audit,binding=binding,groups={})
        a=read('audit.json');window=dict(start_ns=a['window_start_ns'],end_ns=a['window_end_ns']);plan=read('execution_plan.json')
        robot_class={r['robot_id']:c['audit']['id'] for c in plan['classes'] for r in c['robots']}
        for c in plan['classes']:
            for component in c['audit']['components']:groups[c['audit']['id']+'/'+component['id']]=[]
        for r in a['records']:
            if r['record_type']!='request' or r['warmup'] or not window['start_ns']<=r['sent_ns']<window['end_ns']:continue
            status=r['status'] if r['status']!='ok' or r['valid_output'] else 'invalid'
            groups[robot_class[r['robot_id']]+'/'+r['component_id']].append(dict(request_id=r['request_id'],cohort_ns=r['sent_ns'],status=status,
                latency_ms=(r['finished_ns']-r['sent_ns'])/1e6 if status=='ok' else None))
        extra=dict(cohort_definition='non-warmup requests sent in the measurement window; unsent periodic obligations separately retained',
                   periodic_slot_metrics=audit['periodic_slot_metrics'],trial_status=audit['status'],trial_reason=audit['reason'])
    else:raise ValueError('Use profile, pipeline or r05 evidence')
    summaries={k:latency_blocks(rows,start_ns=window['start_ns'],end_ns=window['end_ns'],block_ns=block_ns,seed=seed,resamples=resamples,confidence=confidence) for k,rows in groups.items()}
    return dict(schema_version='1.0',status='estimated' if all(s['status']=='estimated' for s in summaries.values()) else 'insufficient',
        binding=binding,groups=summaries,**extra,automatic_admission=False)
