import pickle
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from rosa_repro.libero_bridge import _load_initial_states


def test_numpy_state_matrix_loads_without_global_allowlist_leak(tmp_path):
    path=tmp_path/'states.pt';states=np.arange(24,dtype=np.float64).reshape(3,8)
    torch.save(states,path)
    before=list(torch.serialization.get_safe_globals())
    np.testing.assert_array_equal(_load_initial_states(path),states)
    assert torch.serialization.get_safe_globals()==before


@pytest.mark.parametrize('states',[np.ones(4),np.full((2,3),np.nan),np.ones((2,3),dtype=np.float32)])
def test_unexpected_state_data_rejected(tmp_path,states):
    path=tmp_path/'states.pt';torch.save(states,path)
    with pytest.raises((ValueError,pickle.UnpicklingError)):
        _load_initial_states(path)


def test_unrelated_pickle_class_is_not_allowed(tmp_path):
    from datetime import datetime
    path=tmp_path/'states.pt';torch.save(datetime(2026,10,31),path)
    with pytest.raises(pickle.UnpicklingError):
        _load_initial_states(path)
