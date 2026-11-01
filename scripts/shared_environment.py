"""Prepare a fresh common GPU environment and bounded memory settings; never installs locally."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rosa_repro.shared_environment import inspect,recipe,export,preflight,memory_policy,bind
from rosa_repro.profile_protocol import template
from rosa_repro.regression import write_json
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
 q=sub.add_parser('prepare');q.add_argument('--repo',required=True);q.add_argument('--environment',required=True);q.add_argument('--uv',default='/usr/local/bin/uv');q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('lock-inputs');q.add_argument('--output',type=Path,required=True)
 for name in ('inspect','preflight'):
  q=sub.add_parser(name);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 if a.op=='lock-inputs':
  from rosa_repro.shared_lock import prepare
  prepare(a.output)
 elif a.op=='prepare':
  value=recipe(repo=a.repo,environment=a.environment,uv=a.uv);export(value,a.output)
  policy=memory_policy(s1_fraction=.35,vlm_fractions=dict(s2=.15,safety=.1,monitor=.15),headroom_fraction=.25)
  write_json(a.output/'probe-bindings.template.json',bind({r:template(r,'groot' if r=='s1' else 'vllm') for r in ('s1','s2','safety','monitor')},policy,python=a.environment+'/bin/python'))
 else:
  value=inspect() if a.op=='inspect' else preflight();write_json(a.output,value)
  if value['status'] in ('blocked','conflict'):raise SystemExit(2)
