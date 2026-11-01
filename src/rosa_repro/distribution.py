"""Offline consistency of shipped entry points; never resolves model credentials."""
import json
from pathlib import Path
from .profile_protocol import template,validate_probe_config
from .execution_plan import digest


def check(root):
    root=Path(root);files={};roles=[]
    for role,engine in [('s1','groot')]+[(r,e) for r in ('s2','safety','monitor') for e in ('transformers','vllm')]:
        name=f'examples/postpod/{role}-{engine}.json';c=json.loads((root/name).read_text())
        validate_probe_config(c,ready=False)
        if c!=template(role,engine):raise ValueError('Shipped config differs from CLI template: '+name)
        try:validate_probe_config(c)
        except ValueError:pass
        else:raise ValueError('Shipped template unexpectedly executable')
        files[name]=digest(c);roles.append(dict(role=role,engine=engine,cuda_measurement=c['binding']['kwargs']['cuda_measurement']))
    for name in ('s1-request.template.json','s1-batch-request.template.json'):
        path='examples/adoption/'+name;q=json.loads((root/path).read_text())['quality']
        required={'regression','paired','state_bank','association','r06'}
        if 'batch-' in name:required|={'batch_regression','batch_diagnostic'}
        if set(q)!=required or set(q['r06'])!={'suite','bank_root','directories'}:raise ValueError('S1/R06 distribution contract missing: '+name)
        from .profile_adoption import inside
        for p in [v for k,v in q.items() if k!='r06']+list(q['r06'].values()):inside(root,p)
        files[path]=digest(q)
    return dict(status='passed',template_count=len(roles),roles=roles,template_hashes=files,models_started=False,network_used=False,
        scope='distributed unresolved templates match current constructors; not GPU readiness or model access verification')
