"""Run analytical cases and independent oracle, saving a new result folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import ortools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rosa_repro.contracts import load_json, validate_manifest
from rosa_repro.domain import Problem
from rosa_repro.measurement import evaluate_p1
from rosa_repro.oracle import feasible, max_frequency_grid
from rosa_repro.scheduler import HomogeneousScheduler, validate_schedule


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    analytical={'s1':2.5,'batch_tradeoff':1/.28,'s1_s2':1/.30,'p2':2.5,'p4':1/.30}
    for name,expected in analytical.items():
        path=ROOT/'examples'/f'{name}.json'
        p=Problem.from_dict(load_json(path))
        result=HomogeneousScheduler(p).solve()
        reference=max_frequency_grid(p,.005)
        validate_schedule(p,result['frequency_hz'],result['placements'],result['routes'])
        passed=(result['search_status']=='converged' and abs(result['frequency_hz']-expected)<=.01
                and abs(result['frequency_hz']-reference)<=.01)
        cases.append({'case':name,'profile_source':'synthetic','expected_analytic_hz':expected,
                      'oracle_grid_hz':reference,'computed_hz':result['frequency_hz'],
                      'absolute_error_hz':abs(result['frequency_hz']-expected),'passed':passed,
                      'input_sha256':sha(path)})
        (args.output/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    adversarial=Problem.from_dict(load_json(ROOT/'examples'/'unresolved_packing.json'))
    check=HomogeneousScheduler(adversarial).check_frequency(2)
    oracle_witness=feasible(adversarial,2)
    recovery=HomogeneousScheduler(adversarial,recover_alternatives=True).check_frequency(2)
    recovery_search=HomogeneousScheduler(adversarial,recover_alternatives=True).solve()
    reference=max_frequency_grid(adversarial,.005)
    validate_schedule(adversarial,2,recovery['placements'],recovery['routes'])
    unresolved={'frequency_hz':2,'scheduler':check,'oracle_finds_feasible_schedule':oracle_witness,
                'recovery_extension':recovery,'recovery_search_hz':recovery_search['frequency_hz'],
                'oracle_grid_hz':reference,
                'interpretation':'Paper-first solver leaves alternate packing unresolved. The explicitly labelled per-robot assignment extension finds a validated witness.',
                'guard_test_passed':check['status']=='unknown' and oracle_witness,
                'extension_test_passed':recovery['status']=='feasible' and abs(recovery_search['frequency_hz']-reference)<=.01}
    (args.output/'unresolved_packing.json').write_text(json.dumps(unresolved,indent=2)+'\n')
    measurement=evaluate_p1(**load_json(ROOT/'examples'/'p1_measurement.json'))
    (args.output/'p1_measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
    source_paths=sorted((ROOT/'src').rglob('*.py'))
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    source_digest=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    manifest={'schema_version':'1.0','run_id':args.output.name,'mode':'cpu_solver',
              'measurement_phase':'exploratory','source_sha256':source_digest,'git_commit':None,
              'git_dirty':None,'git_diff_sha256':None,'dependency_lock_sha256':sha(ROOT/'requirements.lock'),
              'python_version':platform.python_version(),'clock_domain':'local-cpu-verification',
              'gpu_count':0,'gpus':[],'model_runtime':[],
              'initial_states_sha256':None,'observation_manifest_sha256':None,
              'profile_manifest_sha256':None,'unknown_fields':[]}
    validate_manifest(manifest)
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'verification':'R00/R01 initial CPU implementation','platform':platform.platform(),
            'python':platform.python_version(),'ortools':ortools.__version__,
            'gpu_experiments_executed':False,'cases':cases,'known_unresolved_case':unresolved,
            'source_hashes':source_hashes,'all_primary_cases_passed':all(c['passed'] for c in cases),
            'full_R01_general_packing_search_complete':False,
            'not_implemented':['GPU serving','measured profiles','heterogeneous scheduler',
                               'P2-P4 qualified-action evaluator','admission/failover'],
            'recovery_limits':'8 robots and 8 remaining goal GPUs; a conservative positive-witness extension, not a globally complete solver'}
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'primary_cases':len(cases),'all_primary_cases_passed':report['all_primary_cases_passed'],
                      'known_unresolved_case_guard_passed':unresolved['guard_test_passed'],
                      'extension_test_passed':unresolved['extension_test_passed'],
                      'output':str(args.output)},indent=2))
    if not report['all_primary_cases_passed'] or not unresolved['guard_test_passed'] or not unresolved['extension_test_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
