import logging
import unittest
import pytest

logging.basicConfig(level=logging.DEBUG)

import os

os.environ['HPOBENCH_DEBUG'] = 'true'

from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark, \
    NavalPropulsionBenchmark, ParkinsonsTelemonitoringBenchmark, ProteinStructureBenchmark


class TestTabularBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = SliceLocalizationBenchmark(
            rng=1,
            )
        self.default_config = self.benchmark.get_configuration_space(seed=1).get_default_configuration()
        self.socket_id = self.benchmark.socket_id

    def test_tabular_benchmark_wrong_input(self):
        default_config = self.default_config

        benchmark = SliceLocalizationBenchmark(
            socket_id=self.socket_id
        )

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

    def test_slice_benchmark(self):
        default_config = self.default_config

        benchmark = SliceLocalizationBenchmark(
            rng=1,
        )
        result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1),
                                              run_index=[0, 1, 2, 3])

        mean = 0.01828
        assert result['function_value'] == pytest.approx(mean, abs=0.0001)

        runs = result['info']['valid_rmse_per_run']
        calculated_mean = sum(runs) / len(runs)
        assert calculated_mean == pytest.approx(mean, abs=0.0001)

        runtime = 23.1000
        assert result['cost'] == pytest.approx(runtime, abs=0.0001)

        runtimes = sum(result['info']['runtime_per_run'])
        assert runtimes == pytest.approx(runtime, abs=0.0001)

    def test_naval_benchmark(self):
        default_config = self.default_config

        benchmark = NavalPropulsionBenchmark(
            rng=1,
        )
        result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1),
                                              run_index=[0, 1, 2, 3])

        mean = 0.8928
        assert result['function_value'] == pytest.approx(mean, abs=0.0001)

        runs = result['info']['valid_rmse_per_run']
        calculated_mean = sum(runs) / len(runs)
        assert calculated_mean == pytest.approx(mean, abs=0.0001)

        runtime = 5.2477
        assert result['cost'] == pytest.approx(runtime, abs=0.0001)

        runtimes = sum(result['info']['runtime_per_run'])
        assert runtimes == pytest.approx(runtime, abs=0.0001)

    def test_protein_benchmark(self):
        default_config = self.default_config

        benchmark = ProteinStructureBenchmark(
            rng=1,
        )
        result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1),
                                              run_index=[0, 1, 2, 3])

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

    def test_parkinson_benchmark(self):
        default_config = self.default_config

        benchmark = ParkinsonsTelemonitoringBenchmark(
            rng=1,
        )
        result = benchmark.objective_function(configuration=default_config, fidelity=dict(budget=1),
                                              run_index=[0, 1, 2, 3])

        mean = 0.7425
        assert result['function_value'] == pytest.approx(mean, abs=0.0001)

        with pytest.raises(AssertionError):
            benchmark.objective_function_test(default_config, fidelity=dict(budget=1, ))

        result = benchmark.objective_function_test(configuration=default_config, fidelity=dict(budget=100))
        assert pytest.approx(0.15010187, result['function_value'], abs=0.001)

        runtime = 62.7268
        assert result['cost'] == pytest.approx(runtime, abs=0.0001)
