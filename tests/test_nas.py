import pytest

import logging
logging.basicConfig(level=logging.DEBUG)

from hpolib.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark, NavalPropulsionBenchmark, \
    ParkinsonsTelemonitoringBenchmark, ProteinStructureBenchmark, FCNetBenchmark, FCNetBaseBenchmark

try:
    import Pyro4
    skip_container_test = False
except ImportError:
    skip_container_test = True


def setup():
    data_path = '/home/philipp/Dokumente/Code/TabularBenchmarks/fcnet_tabular_benchmarks'
    default_config = FCNetBenchmark.get_configuration_space().get_default_configuration()
    return data_path, default_config


def test_tabular_benchmark_wrong_input():
    data_path, default_config = setup()
    benchmark = SliceLocalizationBenchmark(data_path=data_path)

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, budget=0)

    with pytest.raises(ValueError):
        benchmark.objective_function(configuration=default_config, budget=1, run_index=0.1)

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, budget=0, run_index=[4])

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, budget=0, run_index=[])

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, budget=0, run_index=-1)

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, budget=0, run_index=4)


def test_slice_benchmark():
    data_path, default_config = setup()

    benchmark = SliceLocalizationBenchmark(data_path=data_path)
    result = benchmark.objective_function(configuration=default_config, budget=1, run_index=[0, 1, 2, 3])

    mean = 0.01828
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 5.7750
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = result['info']['runtime_per_run']
    calculated_runtime = sum(runtimes) / len(runtimes)
    assert calculated_runtime == pytest.approx(runtime, abs=0.0001)


def test_naval_benchmark():
    data_path, default_config = setup()

    benchmark = NavalPropulsionBenchmark(data_path=data_path)
    result = benchmark.objective_function(configuration=default_config, budget=1, run_index=[0, 1, 2, 3])

    mean = 0.8928
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 1.3119
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = result['info']['runtime_per_run']
    calculated_runtime = sum(runtimes) / len(runtimes)
    assert calculated_runtime == pytest.approx(runtime, abs=0.0001)


def test_protein_benchmark():
    data_path, default_config = setup()

    benchmark = ProteinStructureBenchmark(data_path=data_path)
    result = benchmark.objective_function(configuration=default_config, budget=1, run_index=[0, 1, 2, 3])

    mean = 0.4474
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 4.8105
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = result['info']['runtime_per_run']
    calculated_runtime = sum(runtimes) / len(runtimes)
    assert calculated_runtime == pytest.approx(runtime, abs=0.0001)


def test_parkinson_benchmark():
    data_path, default_config = setup()

    benchmark = ParkinsonsTelemonitoringBenchmark(data_path=data_path)
    result = benchmark.objective_function(configuration=default_config, budget=1, run_index=[0, 1, 2, 3])

    mean = 0.7425
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 0.6272
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = result['info']['runtime_per_run']
    calculated_runtime = sum(runtimes) / len(runtimes)
    assert calculated_runtime == pytest.approx(runtime, abs=0.0001)
