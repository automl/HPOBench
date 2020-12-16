import logging

import pytest

logging.basicConfig(level=logging.DEBUG)

import os

os.environ['HPOBENCH_DEBUG'] = 'true'

from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark, \
    NavalPropulsionBenchmark, ParkinsonsTelemonitoringBenchmark, ProteinStructureBenchmark


def setup():
    benchmark = SliceLocalizationBenchmark()
    default_config = benchmark.get_configuration_space(seed=1).get_default_configuration()
    return default_config


def test_tabular_benchmark_wrong_input():
    default_config = setup()
    benchmark = SliceLocalizationBenchmark(rng=1)

    with pytest.raises(ValueError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=0))

    with pytest.raises(ValueError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=0.1)

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=[4])

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=[])

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=-1)

    with pytest.raises(AssertionError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=4)

    with pytest.raises(ValueError):
        benchmark.objective_function(configuration=default_config, fidelity=dict(budget=101), run_index=3)

    with pytest.raises((AssertionError, ValueError)):
        benchmark.objective_function_test(configuration=default_config, fidelity=dict(budget=107))

    benchmark = None


def test_slice_benchmark():
    default_config = setup()

    benchmark = SliceLocalizationBenchmark(rng=1)
    result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=[0, 1, 2, 3])

    mean = 0.01828
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 23.1000
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = sum(result['info']['runtime_per_run'])
    assert runtimes == pytest.approx(runtime, abs=0.0001)
    benchmark = None


def test_naval_benchmark():
    default_config = setup()

    benchmark = NavalPropulsionBenchmark(rng=1)
    result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=[0, 1, 2, 3])

    mean = 0.8928
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 5.2477
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = sum(result['info']['runtime_per_run'])
    assert runtimes == pytest.approx(runtime, abs=0.0001)
    benchmark = None


def test_protein_benchmark():
    default_config = setup()

    benchmark = ProteinStructureBenchmark(rng=1)
    result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=[0, 1, 2, 3])

    mean = 0.4474
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    runs = result['info']['valid_rmse_per_run']
    calculated_mean = sum(runs) / len(runs)
    assert calculated_mean == pytest.approx(mean, abs=0.0001)

    runtime = 19.24213
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)

    runtimes = result['info']['runtime_per_run']
    calculated_runtime = sum(runtimes)
    assert calculated_runtime == pytest.approx(runtime, abs=0.0001)
    benchmark = None


def test_parkinson_benchmark():
    default_config = setup()

    benchmark = ParkinsonsTelemonitoringBenchmark(rng=1)
    result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1), run_index=[0, 1, 2, 3])

    mean = 0.7425
    assert result['function_value'] == pytest.approx(mean, abs=0.0001)

    with pytest.raises(AssertionError):
        benchmark.objective_function_test(default_config, fidelity=dict(budget=1,))

    result = benchmark.objective_function_test(configuration=default_config, fidelity=dict(budget=100))
    assert pytest.approx(0.15010187, result['function_value'], abs=0.001)

    runtime = 62.7268
    assert result['cost'] == pytest.approx(runtime, abs=0.0001)
    benchmark = None
