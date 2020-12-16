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

    from hpobench.container.benchmarks.surrogates.paramnet_benchmark import ParamNetHiggsOnStepsBenchmark

    benchmark = ParamNetHiggsOnStepsBenchmark()
    cs = benchmark.get_configuration_space(0)
    fs = benchmark.get_fidelity_space(0)
    cfg = cs.sample_configuration()

    result = benchmark.objective_function(cfg)
    assert result['function_value'] == pytest.approx(0.3244, 0.01)

    full_budget = fs.get_default_configuration()
    assert full_budget['step'] == 50

    result_on_full_budget = benchmark.objective_function(configuration=cfg, fidelity=full_budget)
    assert result['function_value'] == pytest.approx(result_on_full_budget['function_value'], abs=0.000001)

    result_on_small_budget = benchmark.objective_function(configuration=cfg, fidelity={'step': 1})
    assert result['cost'] == pytest.approx(result_on_small_budget['cost'] * 50, abs=0.0001)


def test_param_net_time():
    from hpobench.container.benchmarks.surrogates.paramnet_benchmark import ParamNetHiggsOnTimeBenchmark

    benchmark = ParamNetHiggsOnTimeBenchmark()
    cs = benchmark.get_configuration_space(0)
    cfg = cs.sample_configuration()

    result = benchmark.objective_function(cfg)
    assert result['function_value'] == pytest.approx(0.3244, 0.01)
    assert isinstance(result['info']['learning_curve'], list)

    assert result['info']['observed_epochs'] == 193
    assert result['info']['learning_curve'][49] == result['info']['learning_curve'][192]

    # Test the case when the budget is less than the costs for 50 epochs
    fidelity = {'budget': 50}
    result = benchmark.objective_function(configuration=cfg, fidelity=fidelity)
    assert result['info']['observed_epochs'] == 38
    assert len(result['info']['learning_curve']) == 38
