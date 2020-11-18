import pytest

# import logging
# logging.basicConfig(level=logging.DEBUG)
# from hpobench.util.container_utils import enable_container_debug
# enable_container_debug()


def test_load_data():
    from hpobench.util.data_manager import ParamNetDataManager

    with pytest.raises(AssertionError):
        dm = ParamNetDataManager(dataset='unknown_dataset')

    dm = ParamNetDataManager(dataset='higgs')
    obj_v_fn, costs_fn = dm.load()

    assert obj_v_fn is not None
    assert costs_fn is not None


def test_obj_func():

    from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamnetBenchmark

    benchmark = ParamnetBenchmark(dataset='higgs')
    cs = benchmark.get_configuration_space(0)
    fs = benchmark.get_fidelity_space(0)
    cfg = cs.sample_configuration()

    result = benchmark.objective_function(cfg)
    assert result['function_value'] == pytest.approx(0.3244, 0.01)


def test_param_net_time():
    from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamnetTimeBenchmark
    benchmark = ParamnetTimeBenchmark(dataset='higgs')
    cs = benchmark.get_configuration_space(0)
    fs = benchmark.get_fidelity_space(0)
    cfg = cs.sample_configuration()

    result = benchmark.objective_function(cfg)
    assert result['function_value'] == pytest.approx(0.3244, 0.01)
    assert isinstance(result['info']['learning_curve'], list)
