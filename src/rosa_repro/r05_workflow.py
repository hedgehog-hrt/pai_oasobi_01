"""R05 prerequisite audit and prospective comparison, preserving incomplete cells."""
from pathlib import Path
import json,statistics
from .profile_adoption import inside
from .execution_plan import digest
from .r05_runtime import reaudit

read=lambda p:json.loads(Path(p).read_text())


def check_pilots(s,root,manifest):
    if s['stage']=='pilot':
        if manifest:raise ValueError('Pilot has no predecessor dependency')
        return {}
    if not isinstance(manifest,dict) or set(manifest)!=set(s['groups']):raise ValueError('Every confirmation group needs five completed pilot conditions')
    from .r05_prediction import verify as verify_prediction
    for text in s['prediction_artifacts'].values():
        pred=verify_prediction(json.loads(text),root)
        for group in set(pred['groups']) & set(s['groups']):
            if sorted(pred['inputs']['pilot_directories'][group])!=sorted(manifest[group]):raise ValueError('Prediction and confirmation pilot dependencies differ')
    evidence={};allpaths=set()
    for group in s['groups']:
        rows=[];batches=[];runtime=None
        for path in manifest[group]:
            p=inside(root,path)
            if p.resolve() in allpaths:raise ValueError('Pilot evidence reused')
            allpaths.add(p.resolve());a=reaudit(p);prior=read(p/'suite.json');cell=a['cell']
            if a['status']!='complete' or prior['stage']!='pilot' or cell['group']!=group:raise ValueError('Completed matching R05 pilot required')
            if any(prior[k]!=s[k] for k in ('config','corpus','split','source_manifest')):raise ValueError('Pilot runtime/input/source binding differs')
            if runtime is not None and a['backend_attestation']!=runtime:raise ValueError('Pilot workers have different observed runtimes')
            runtime=a['backend_attestation'];batches.append(cell['batch_size']);rows.append(dict(directory=path,evidence_sha256=a['evidence_sha256']))
        if sorted(batches)!=[1,2,4,8,16]:raise ValueError('Pilot group must cover exactly five batch caps')
        evidence[group]=dict(rows=rows,backend_attestation=runtime)
    return evidence


def report(s,root,directories):
    if s['stage']=='confirmation':
        manifest={}
        for text in s['prediction_artifacts'].values():
            prediction=json.loads(text)
            for group,paths in prediction['inputs']['pilot_directories'].items():
                if group in s['groups']:
                    if group in manifest and manifest[group]!=paths:raise ValueError('Conflicting prediction dependencies')
                    manifest[group]=paths
        check_pilots(s,root,manifest)
    if set(directories)!={c['id'] for c in s['cells']} or len(set(directories.values()))!=len(directories):raise ValueError('Prospective directory map must cover every cell once')
    rows=[];runs=set();runtime=None
    for cell in s['cells']:
        path=inside(root,directories[cell['id']]);row=dict(cell=cell,directory=directories[cell['id']])
        if not path.exists():row.update(status='missing',reason='expected_directory_absent')
        else:
            try:
                a=reaudit(path);r=read(path/'result.json')
                if a['suite_id']!=s['suite_id'] or a['cell']!=cell:raise ValueError('Trial suite/cell differs')
                if r['run_id'] in runs:raise ValueError('Run reused across comparison cells')
                runs.add(r['run_id'])
                if runtime is not None and a['backend_attestation']!=runtime:raise ValueError('Observed runtimes differ')
                runtime=a['backend_attestation'];row.update(a)
            except (ValueError,KeyError,TypeError,OSError) as exc:row.update(status='invalid',reason=str(exc))
        rows.append(row)
    def summary(values):return dict(n=len(values),mean=statistics.mean(values) if values else None,sample_stddev=statistics.stdev(values) if len(values)>1 else None)
    groups=[]
    for group in s['groups']:
        selected=[r for r in rows if r['cell']['group']==group];successful=[r for r in selected if r['status']=='complete']
        g=dict(group=group,status_counts=dict(__import__('collections').Counter(r['status'] for r in selected)),
            complete=len(selected)==len(successful),batch_summaries={},paired_speedup=None)
        for batch in sorted({r['cell']['batch_size'] for r in selected}):
            values=[r['metrics']['measurement']['completed_virtual_action_rate_hz'] for r in successful if r['cell']['batch_size']==batch]
            g['batch_summaries'][str(batch)]=summary(values)
        if s['stage']=='confirmation' and g['complete']:
            ratios=[];pairs=[]
            for rep in sorted({r['cell']['repeat'] for r in selected}):
                baseline=next(r for r in selected if r['cell']['repeat']==rep and r['cell'].get('candidate')=='batch_one')
                candidate=baseline if 'predicted_best' in baseline['cell']['aliases'] else next(r for r in selected if r['cell']['repeat']==rep and r['cell'].get('candidate')=='predicted_best')
                b=baseline['metrics']['measurement']['completed_virtual_action_rate_hz'];v=candidate['metrics']['measurement']['completed_virtual_action_rate_hz']
                ratio=v/b if b>0 else None
                pairs.append(dict(repeat=rep,baseline_cell=baseline['cell']['id'],candidate_cell=candidate['cell']['id'],speedup=ratio))
                if ratio is not None:ratios.append(ratio)
            g['paired_speedup']=dict(pairs=pairs,summary=summary(ratios),complete=len(ratios)==len(pairs),
                uncertainty_unit='independent confirmation repetitions; no request-level confidence interval')
        groups.append(g)
    return dict(schema_version='1.0',suite_id=s['suite_id'],mode=s['config']['mode'],stage=s['stage'],rows=rows,groups=groups,
        status='complete' if all(g['complete'] for g in groups) else 'incomplete',latency_slo_ms=None,
        metric='completed_virtual_action_chunks_per_second',paper_reproduction_confirmed=False,
        limitations=['One GPU adaptation; paper Fig.14 worker count unresolved. Saved observations and virtual action waits are not LIBERO task success.',
          'Batch caps can exceed robot count. Reported actual batch sizes, including partial batches, determine observed execution.',
          'Ratios compare independent confirmation runs only. Failed/missing cells suppress group speedup; no unmeasured optimum is claimed.',
          'Operational watchdogs bound execution. A timeout invalidates a comparative trial; it is not a latency-SLO failure.'])
