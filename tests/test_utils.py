import numpy as np
import pytest


def test_example_utils():
    from hpobench.util.example_utils import get_travis_settings

    res = get_travis_settings('smac')
    assert res['runcount-limit'] == 5

    res = get_travis_settings('bohb')
    assert res['max_budget'] == 2

    with pytest.raises(ValueError):
        res = get_travis_settings('unknown')


def test_example_utils_2():
    from hpobench.util.example_utils import set_env_variables_to_use_only_one_core
    import os
    set_env_variables_to_use_only_one_core()
    assert os.environ['OMP_NUM_THREADS'] == '1'
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
    assert os.environ['MKL_NUM_THREADS'] == '1'
    assert os.environ['VECLIB_MAXIMUM_THREADS'] == '1'
    assert os.environ['NUMEXPR_NUM_THREADS'] == '1'
    assert os.environ['NUMEXPR_MAX_THREADS'] == '1'


def test_rng_helper():
    from hpobench.util.rng_helper import _cast_int_to_random_state

    rng = np.random.RandomState(123)

    with pytest.raises(ValueError):
        _cast_int_to_random_state('not_an_int')

    assert rng == _cast_int_to_random_state(rng)

    rng = np.random.RandomState(123)
    assert rng.random() == _cast_int_to_random_state(123).random()


def test_rng_helper_2():
    from hpobench.util.rng_helper import get_rng

    rng = get_rng(None, None)
    assert isinstance(rng, np.random.RandomState)

    old_rng = np.random.RandomState(123)
    rng = get_rng(None, old_rng)
    assert rng == old_rng


def test_debug_level():
    from hpobench.util.container_utils import enable_container_debug, disable_container_debug
    import os
    enable_container_debug()
    assert os.environ['HPOBENCH_DEBUG'] == 'true'

    disable_container_debug()
    assert os.environ['HPOBENCH_DEBUG'] == 'false'
