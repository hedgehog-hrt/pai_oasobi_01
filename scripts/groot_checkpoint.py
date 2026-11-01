"""Inspect the trusted GR00T pin or hash an already downloaded checkpoint offline."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.checkpoint_verification import groot_pin,verify_groot_checkpoint
from rosa_repro.regression import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['describe','verify'])
    p.add_argument('--model-path',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.command=='describe':write_json(a.output,groot_pin());return 0
    if a.model_path is None:p.error('--model-path required')
    try:report=verify_groot_checkpoint(a.model_path)
    except Exception as exc:
        write_json(a.output,dict(status='failed',error_type=type(exc).__name__,message=str(exc),full_weight_hashes_verified=False));return 1
    write_json(a.output,report);return 0


if __name__=='__main__':raise SystemExit(main())
