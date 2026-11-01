import json,shutil,subprocess,sys
from pathlib import Path
import pytest
from rosa_repro.distribution import check
ROOT=Path(__file__).resolve().parents[1]


def test_distribution_and_actual_cli_template_match(tmp_path):
 r=check(ROOT);assert r['template_count']==7 and not r['models_started']
 for role,engine in [('s1','groot'),('safety','transformers'),('monitor','vllm')]:
  p=tmp_path/(role+'.json')
  subprocess.run([sys.executable,str(ROOT/'scripts/postpod_models.py'),'template','--role',role,'--engine',engine,'--output',str(p)],check=True,capture_output=True)
  assert json.loads(p.read_text())==json.loads((ROOT/f'examples/postpod/{role}-{engine}.json').read_text())


@pytest.mark.parametrize('drift',['measurement','r06'])
def test_old_distribution_drift_is_detected(tmp_path,drift):
 shutil.copytree(ROOT/'examples',tmp_path/'examples')
 p=tmp_path/('examples/postpod/s1-groot.json' if drift=='measurement' else 'examples/adoption/s1-request.template.json')
 d=json.loads(p.read_text())
 if drift=='measurement':del d['binding']['kwargs']['cuda_measurement']
 else:del d['quality']['r06']
 p.write_text(json.dumps(d))
 with pytest.raises(ValueError):check(tmp_path)
