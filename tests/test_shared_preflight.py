"""Exercise the real model-preflight entry point with explicit fake metadata/weights."""
import json,sys
from pathlib import Path
import pytest
from rosa_repro.shared_environment import VARIANT,bind,memory_policy
from rosa_repro.profile_protocol import template,preflight
from rosa_repro.regression import write_json
from rosa_repro.resume_workflow import command,validate_output


def config(role,shared):
    configs={r:template(r,'groot' if r=='s1' else 'vllm') for r in ('s1','s2','safety','monitor')}
    if shared:
        policy=memory_policy(s1_fraction=.35,vlm_fractions=dict(s2=.15,safety=.1,monitor=.15),headroom_fraction=.25)
        configs=bind(configs,policy,python=str(Path(sys.executable).resolve()))['configs']
    c=configs[role];c['hardware_kind']='SYNTHETIC_METADATA_TEST_NO_GPU';c['binding']['kwargs']['model_path']='/tmp/rosa-metadata-test-no-weights'
    return c


@pytest.mark.parametrize('role',['s2','safety','monitor'])
@pytest.mark.parametrize('shared',[False,True])
def test_resume_model_command_uses_matching_variant(role,shared,tmp_path,monkeypatch):
    # This covers resume_workflow -> postpod_models CLI -> preflight -> version
    # check, without importing a GPU/model loader or touching checkpoints.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from postpod_models import main
    c=config(role,shared);path=tmp_path/'config.json';write_json(path,c);out=tmp_path/'stage';out.mkdir()
    versions={'torch':'2.9.0+cu128','transformers':'4.57.3' if shared else '4.57.6','Pillow':'12.0.0','vllm':'0.11.2'}
    monkeypatch.setattr('rosa_repro.vlm_backends.version',lambda name:versions[name])
    monkeypatch.setattr('rosa_repro.profile_protocol.environment_evidence',lambda:dict(synthetic_metadata_fixture=True))
    checked=[]
    def snapshot(path,model_id):checked.append((path,model_id));return dict(synthetic_checkpoint_fixture=True)
    monkeypatch.setattr('rosa_repro.vlm_contract.verify_snapshot',snapshot)
    p=dict(config=str(path),worker_python=sys.executable);step=dict(kind='model')
    argv,env=command(p,step,out,{},None);assert argv[0]==sys.executable and argv[2]=='preflight'
    main(argv[2:]);r=json.loads((out/'preflight.json').read_text())
    assert validate_output(p,step,out)=={} and r['status']=='passed' and not r['gpu_model_loaded']
    assert r['runtime_variant']==(VARIANT if shared else None) and len(checked)==1
    versions['transformers']='4.57.6' if shared else '4.57.3'
    with pytest.raises(ValueError,match='versions differ'):preflight(c)
    assert len(checked)==1  # Mismatch stops before reading weights.


def test_unknown_variant_stops_before_environment_probe(monkeypatch):
    c=config('s2',True);c['binding']['kwargs']['runtime_variant']='unreviewed'
    monkeypatch.setattr('rosa_repro.profile_protocol.environment_evidence',lambda:pytest.fail('Invalid binding must fail before probing'))
    with pytest.raises(ValueError,match='runtime'):preflight(c)
