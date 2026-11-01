"""Prepare every variant/repetition and bounded runner commands; starts no models."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.profile_adoption import read
from rosa_repro.regression import write_json
from rosa_repro.dedicated_comparison import prepare_family


def export(value,out,root):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);jobs=[];paths={}
    write_json(out/'family.json',value);write_json(out/'paper.json',value['paper']);write_json(out/'reference.json',value['reference'])
    for row in value['rows']:
        cell=row['cell'];cid=cell['id'];folder=out/cid;folder.mkdir();write_json(folder/'condition.json',cell)
        raw=folder/'runs'/f"closed-loop-{cell['repeat']:03d}"/'attempt-001/trial/raw'
        paths[cid]=str(raw.resolve().relative_to(Path(root).resolve()))
        if row['status']=='resource_infeasible':
            write_json(folder/'resources.json',row['resources']);jobs.append(dict(cell_id=cid,status=row['status'],argv=None));continue
        write_json(folder/'plan.json',row['plan']);write_json(folder/'suite.json',row['suite'])
        cmd=[sys.executable,str(ROOT/'scripts/pipeline_experiments.py'),'run','--suite',str((folder/'suite.json').resolve()),
            '--root',str(Path(root).resolve()),'--output',str((folder/'runs').resolve()),'--case-id',f"closed-loop-{cell['repeat']:03d}",
            '--max-new','1','--budget-seconds',str(row['suite']['trial_wall_budget_s']+30),'--report',str((folder/'dry-run-report.json').resolve())]
        execute_cmd=cmd[:-1]+[str((folder/'execution-report.json').resolve()),'--execute']
        jobs.append(dict(cell_id=cid,status='dry_run_only',argv=cmd,execute_flag='--execute',
            execute_argv=execute_cmd,
            GPU_required_options=(['--address','--adoptions','--sim-python','--sim-repo']+(['--joint-review'] if row['plan'].get('joint_residency_review_required') else [])) if row['plan']['mode']=='gpu' else [],
            joint_review_required=row['plan'].get('joint_residency_review_required',False),
            retry='Use the actual successful attempt path; never replace failed evidence or silently pick a favorable repetition.'))
    order=value['paper']['randomized_order'];jobs.sort(key=lambda j:order.index(j['cell_id']))
    write_json(out/'jobs.json',jobs);write_json(out/'directories.json',paths)
    write_json(out/'references.json',{value['family']:str((out/'reference.json').resolve().relative_to(Path(root).resolve()))})
    return dict(status='prepared',cells=len(jobs),models_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('paper','reference','profiles','banks','root','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--family',required=True);p.add_argument('--phase',choices=['formal','cpu_smoke'],default='formal');p.add_argument('--controls',type=Path)
    p.add_argument('--setup-budget',type=float,default=600)
    a=p.parse_args();a.output.resolve().relative_to(a.root.resolve())
    value=prepare_family(read(a.reference),read(a.paper),a.family,read(a.profiles),read(a.banks),a.root,
        phase=a.phase,controls=read(a.controls) if a.controls else None,setup_budget_s=a.setup_budget)
    print(export(value,a.output,a.root))
