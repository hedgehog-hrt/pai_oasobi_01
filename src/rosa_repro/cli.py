from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from jsonschema.exceptions import ValidationError

from .contracts import load_json
from .domain import Problem
from .measurement import evaluate_p1
from .scheduler import HomogeneousScheduler


def main(argv=None):
    parser = argparse.ArgumentParser(description='ROSA CPU scheduling and event evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    solve = sub.add_parser('solve')
    solve.add_argument('input')
    solve.add_argument('--frequency', type=float)
    solve.add_argument('--recover-alternatives', action='store_true',
                       help='Enable explicitly labelled small-instance assignment extension')
    solve.add_argument('--output')
    measure = sub.add_parser('evaluate-p1')
    measure.add_argument('input', help='JSON object with records and explicit window/context')
    measure.add_argument('--output')
    multi = sub.add_parser('evaluate-pipeline')
    multi.add_argument('input', help='Version 2 P1-P4 event audit')
    multi.add_argument('--output')
    heterogeneous = sub.add_parser('solve-heterogeneous')
    heterogeneous.add_argument('input', help='Heterogeneous fleet/profile JSON')
    heterogeneous.add_argument('--rates', help='JSON file mapping every class ID to a fixed rate in Hz')
    heterogeneous.add_argument('--output')
    oracle = sub.add_parser('oracle-heterogeneous')
    oracle.add_argument('input', help='Small heterogeneous fleet/profile JSON')
    oracle.add_argument('--grid', required=True, help='JSON file mapping class IDs to explicit frequency lists')
    oracle.add_argument('--time-limit', type=float, default=15)
    oracle.add_argument('--output')
    args = parser.parse_args(argv)
    try:
        data = load_json(args.input)
        if args.command == 'solve':
            problem = Problem.from_dict(data)
            scheduler = HomogeneousScheduler(problem,recover_alternatives=args.recover_alternatives)
            result = scheduler.solve() if args.frequency is None else scheduler.check_frequency(args.frequency)
        elif args.command == 'evaluate-pipeline':
            from .multicomponent import evaluate_multicomponent
            result = evaluate_multicomponent(data)
        elif args.command in ('solve-heterogeneous', 'oracle-heterogeneous'):
            from .heterogeneous import Fleet, HeterogeneousScheduler
            fleet = Fleet.from_dict(data)
            if args.command == 'solve-heterogeneous':
                scheduler = HeterogeneousScheduler(fleet)
                result = scheduler.check_rates(load_json(args.rates)) if args.rates else scheduler.solve()
                if args.rates:
                    result['rates_sha256'] = hashlib.sha256(Path(args.rates).read_bytes()).hexdigest()
            else:
                from .heterogeneous_oracle import exhaustive_grid
                result = exhaustive_grid(data, load_json(args.grid), time_limit_s=args.time_limit)
                result['grid_sha256'] = hashlib.sha256(Path(args.grid).read_bytes()).hexdigest()
        else:
            result = evaluate_p1(**data)
        result['input_sha256'] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        if args.output:
            # Refuse to destroy earlier experiment output.
            with Path(args.output).open('x') as f:
                f.write(encoded)
        else:
            print(encoded, end='')
        return 0
    except (ValueError, ValidationError, OSError, TypeError) as exc:
        print(json.dumps({'status': 'invalid_input_or_output', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
