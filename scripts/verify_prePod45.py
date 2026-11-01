"""Repeatable CPU verification of P2-P4 audits and R07, with hashed evidence.
Uses only saved synthetic examples. Does not connect to a Pod or run models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path
import ortools

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rosa_repro import __version__
from rosa_repro.contracts import load_json
from rosa_repro.multicomponent import evaluate_multicomponent
from rosa_repro.heterogeneous import Fleet,HeterogeneousScheduler,validate_fleet_schedule
from rosa_repro.heterogeneous_oracle import exhaustive_grid


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[];inputs={}
    def read(name):
        path=ROOT/'examples'/f'{name}.json';inputs[str(path.relative_to(ROOT))]=sha(path)
        return load_json(path)
    def save(name,data):
        with (args.output/f'{name}.json').open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    for name,expected in [('p2_audit',2),('p3_audit',2),('p4_audit',10),
                          ('p2_late_slot_audit',0),('p3_missing_safety_audit',0),('p4_horizon_audit',10)]:
        result=evaluate_multicomponent(read(name));save(name,result)
        cases.append(dict(case=name,expected_qualified_chunks=expected,qualified_chunks=result['qualified_action_chunks'],
                          weighted_Q=result['weighted_Q'],passed=result['qualified_action_chunks']==expected))
    for name in ('r07_mix2','r07_mix4'):
        data=read(name);rates=read(name+'_rates');fleet=Fleet.from_dict(data)
        fixed=HeterogeneousScheduler(fleet).check_rates(rates)
        if fixed['status']=='feasible':validate_fleet_schedule(fleet,rates,fixed['placements'],fixed['routes'])
        oracle=exhaustive_grid(data,{cid:[rate] for cid,rate in rates.items()})
        search=HeterogeneousScheduler(fleet).solve()
        if search['status']=='feasible':validate_fleet_schedule(fleet,search['frequencies_hz'],search['placements'],search['routes'])
        save(name,dict(fixed_rates=fixed,independent_oracle=oracle,adaptive_search=search))
        passed=(fixed['status']=='feasible' and len(fixed['placements'])==4 and oracle['status']=='grid_optimal'
                and fixed['weighted_objective']==oracle['best']['weighted_objective'] and search['status']=='feasible')
        cases.append(dict(case=name,passed=passed,workers=len(fixed.get('placements',[])),
                          fixed_rate_weighted_objective=fixed.get('weighted_objective'),search_lower_bound=search.get('weighted_objective')))
    data=read('r07_weighted');fleet=Fleet.from_dict(data)
    result=HeterogeneousScheduler(fleet).solve();oracle=exhaustive_grid(data,read('r07_weighted_grid'))
    validate_fleet_schedule(fleet,result['frequencies_hz'],result['placements'],result['routes'])
    gap=7-result['weighted_objective']
    save('r07_weighted',dict(adaptive_search=result,independent_oracle=oracle,analytic_optimum=7,analytic_gap=gap))
    cases.append(dict(case='r07_weighted',passed=0<=gap<=.01 and oracle['best']['weighted_objective']==7,
                      weighted_objective=result['weighted_objective'],analytic_optimum=7,analytic_gap=gap))
    data=read('r07_greedy_counterexample');rates=read('r07_greedy_counterexample_rates')
    result=HeterogeneousScheduler(Fleet.from_dict(data)).check_rates(rates)
    oracle=exhaustive_grid(data,read('r07_greedy_counterexample_grid'))
    save('r07_greedy_counterexample',dict(scheduler=result,independent_oracle=oracle,
         interpretation='The greedy 2+2 merge blocks a valid 4+2 / 3+2 packing. Unknown is intentional; infeasibility would be false.'))
    cases.append(dict(case='r07_greedy_counterexample',passed=result['status']=='unknown' and oracle['status']=='grid_optimal',
                      scheduler_status=result['status'],oracle_status=oracle['status'],oracle_objective=oracle['best']['weighted_objective']))
    paths=sorted([*(ROOT/'src').rglob('*.py'),*(ROOT/'scripts').glob('*.py'),*(ROOT/'tests').glob('*.py'),
                  *(ROOT/'schemas').glob('*.json'),ROOT/'pyproject.toml',ROOT/'requirements.lock'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(schema_version='1.0',verification='P2-P4 evaluator + R07 pre-Pod CPU',package_version=__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),platform=platform.platform(),
        ortools=ortools.__version__,gpu_experiments_executed=False,input_sources='synthetic only',
        cases=cases,all_cases_passed=all(c['passed'] for c in cases),input_hashes=inputs,source_hashes=hashes,
        source_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        limitations=['No GPU or real-model execution','Slot/action association is a declared project convention',
                     'R07 is a heuristic lower-bound search, not a global optimum certificate',
                     'Oracle optimum applies only to its explicit finite frequency grid',
                     'Minimum obligation footprint is not globally proved by greedy compaction',
                     'Live P2-P4 event producers and multi-GPU deployment remain unverified'])
    save('verification',report)
    print(json.dumps(dict(output=str(args.output),cases=len(cases),all_cases_passed=report['all_cases_passed']),indent=2))
    return 0 if report['all_cases_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
