from copy import deepcopy
import pytest
from runtime_helpers import plan_for
from rosa_repro.dedicated_baselines import prepare,validate


def test_physical_count_no_dropped_robots_or_fake_colocation():
 p=plan_for(('P4','P1'));profiles={w['component_id']:w['profile_id'] for w in p['workers']}
 one=prepare(p,'gpu_per_robot',profiles,gpu_budget=2)
 assert one['required_physical_GPUs']==2 and len(one['all_robots'])==2 and not one['executable']
 assert 'joint_model_memory_and_latency_measurement_required' in one['blocking_reasons']
 many=prepare(p,'gpu_per_model_per_robot',profiles,gpu_budget=2)
 assert many['required_physical_GPUs']==5 and many['shortfall']==3 and not many['robot_subset_dropped']
 assert all(len(a['models'])==1 for a in many['assignments'])
 assert validate(many)==many
 bad=deepcopy(many);bad['executable']=True
 with pytest.raises(ValueError):validate(bad)
