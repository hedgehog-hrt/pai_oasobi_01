import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('safety_stop_report',Path(__file__).resolve().parents[1]/'scripts/safety_stop_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def counts(n,stopped,applied):
    return dict(decisions=n,safety_stopped_decisions=stopped,raw_action_chunks=applied)

def test_goal_requires_each_pipeline_to_act_and_enough_decisions():
    assert module.assess_class(counts(100,90,10))['target_met']
    assert not module.assess_class(counts(100,91,9))['target_met']
    assert not module.assess_class(counts(100,0,0))['target_met']
    assert not module.assess_class(counts(10,0,10))['target_met']
    assert module.assess_class(counts(0,0,0))['stop_rate'] is None

def test_stopped_action_cannot_also_count_as_applied():
    with pytest.raises(ValueError):module.assess_class(counts(100,90,11))
