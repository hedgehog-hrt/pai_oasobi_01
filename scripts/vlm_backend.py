"""Prepare pinned VLM configs/processor files or verify an already saved snapshot.
This CLI never downloads model weights, starts a Pod, or runs a GPU model.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rosa_repro.vlm_contract import default_config,pins,verify_snapshot,runtime_fingerprint
from rosa_repro.regression import write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('config');c.add_argument('--engine',choices=['transformers','vllm'],required=True)
    c.add_argument('--output',type=Path,required=True)
    d=sub.add_parser('download-processors');d.add_argument('--model',choices=['3b','7b'],required=True);d.add_argument('--output',type=Path,required=True)
    v=sub.add_parser('verify-snapshot');v.add_argument('--model',choices=['3b','7b'],required=True);v.add_argument('--directory',type=Path,required=True)
    v.add_argument('--processors-only',action='store_true');v.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('bindings');b.add_argument('--engine',choices=['transformers','vllm'],required=True)
    b.add_argument('--snapshot-3b',type=Path,required=True);b.add_argument('--snapshot-7b',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True)
    b.add_argument('--python-executable',help='Optional absolute Python path on every assigned Ray node')
    a=p.parse_args(argv)
    if a.command in ('config','bindings'):
        configs={role:default_config(role,a.engine) for role in ('s2','safety','monitor')}
        if a.command=='config':write_json(a.output,configs)
        else:
            factory='rosa_repro.vlm_backends:'+('transformers_vlm_backend' if a.engine=='transformers' else 'vllm_vlm_backend')
            bindings={role:dict(factory=factory,kwargs=dict(config=c,model_path=str((a.snapshot_3b if role=='safety' else a.snapshot_7b).resolve()))) for role,c in configs.items()}
            if a.python_executable:
                import re
                if not re.fullmatch(r'/[A-Za-z0-9_./-]+',a.python_executable):p.error('Python executable must be a plain absolute path')
                for binding in bindings.values():binding['runtime_env']=dict(py_executable=a.python_executable)
            write_json(a.output,bindings)
        return 0
    mid='Qwen/Qwen2.5-VL-'+('3B' if a.model=='3b' else '7B')+'-Instruct';pin=pins()[mid]
    if a.command=='download-processors':
        from huggingface_hub import snapshot_download
        a.output.mkdir(parents=True,exist_ok=False)
        files=[n for n,row in pin['files'].items() if row['kind']!='weight']
        snapshot_download(mid,revision=pin['revision'],local_dir=str(a.output),allow_patterns=files,token=False)
        evidence=verify_snapshot(a.output,mid,weights=False)
        write_json(a.output/'processor_verification.json',evidence)
        print('Verified processor/config files only; weights were not requested.')
    else:write_json(a.output,verify_snapshot(a.directory,mid,weights=not a.processors_only))
    return 0


if __name__=='__main__':raise SystemExit(main())
