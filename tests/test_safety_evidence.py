import json
import pytest
from rosa_repro.vlm_contract import default_config,validate_config,generation_schema,runtime_fingerprint
from rosa_repro.vlm_backends import parsed_result

def config():
    c=default_config('safety','vllm');c.update(structured_output='safety_evidence_v1',max_new_tokens=64)
    return validate_config(c)

@pytest.mark.parametrize('label,stopped',[('safe',False),('unsafe',True),('unknown',True)])
def test_evidence_does_not_override_verdict(label,stopped):
    c=config();raw=dict(text=json.dumps({'reason':'No person is visible','decision':label}),finish_reason='stop',generated_tokens=20)
    result=parsed_result(c,raw,{})
    assert result.output==dict(stop=stopped,reason='No person is visible')
    assert result.evidence['parser']=='strict-safety-evidence-v1'

@pytest.mark.parametrize('text,finish',[
    ('{"reason":"clear","decision":"safe"}','length'),
    ('{"reason":"","decision":"safe"}','stop'),
    ('{"reason":"clear","decision":"safe","decision":"unknown"}','stop'),
    ('{"decision":"safe"}','stop'),
    ('{"reason":"clear","decision":"safe","extra":"yes"}','stop'),
    (json.dumps({'reason':'x'*193,'decision':'safe'}),'stop')])
def test_bad_evidence_never_clears(text,finish):
    r=parsed_result(config(),dict(text=text,finish_reason=finish,generated_tokens=64),{})
    assert r.output is None and r.error_type=='InvalidModelGeneration'

def test_schema_order_fingerprint_and_role_binding():
    c=config();schema=generation_schema('safety',c['structured_output'])
    assert list(schema['properties'])==schema['required']==['reason','decision']
    assert schema['properties']['reason']=={'type':'string'}
    old=dict(c,structured_output='safety_label_v1')
    assert runtime_fingerprint(old)!=runtime_fingerprint(c)
    wrong=default_config('monitor','vllm');wrong['structured_output']='safety_evidence_v1'
    with pytest.raises(ValueError):validate_config(wrong)
