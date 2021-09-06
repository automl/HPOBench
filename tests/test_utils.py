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


def test_rng_serialization():
    from hpobench.util.rng_helper import deserialize_random_state, serialize_random_state
    rs_old = np.random.RandomState(1)
    r_str = serialize_random_state(rs_old)
    rs_new = deserialize_random_state(r_str)

    assert np.array_equiv(rs_old.random_sample(10), rs_new.random_sample(10))


def test_rng_serialization_xgb():
    import json
    from hpobench.util.container_utils import BenchmarkEncoder, BenchmarkDecoder
    from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark

    b = XGBoostBenchmark(task_id=167149, rng=0)
    meta = b.get_meta_information()

    meta_str = json.dumps(meta, indent=None, cls=BenchmarkEncoder)
    meta_new = json.loads(meta_str, cls=BenchmarkDecoder)
    assert isinstance(meta_new['initial random seed'], np.random.RandomState)
    assert np.array_equiv(meta['initial random seed'].random(10), meta_new['initial random seed'].random(10))


def test_benchmark_encoder():
    from enum import Enum
    class test_enum(Enum):
        obj = 'name'

        def __str__(self):
            return str(self.value)

    from hpobench.util.container_utils import BenchmarkEncoder, BenchmarkDecoder
    import json
    import numpy as np

    enum_obj = test_enum.obj
    enum_obj_str = json.dumps(enum_obj, cls=BenchmarkEncoder)
    assert enum_obj_str == '"name"'

    array = np.array([1, 2, 3, 4])
    array_str = json.dumps(array, cls=BenchmarkEncoder)
    array_ = json.loads(array_str, cls=BenchmarkDecoder)
    assert np.array_equiv(array, array_)


def test_debug_level():
    from hpobench.util.container_utils import enable_container_debug, disable_container_debug
    import os
    enable_container_debug()
    assert os.environ['HPOBENCH_DEBUG'] == 'true'

    disable_container_debug()
    assert os.environ['HPOBENCH_DEBUG'] == 'false'
