import logging

import numpy as np
import pytest

logging.basicConfig(level=logging.DEBUG)

try:
    import Pyro4

    skip_container_test = False
except ImportError:
    skip_container_test = True


def test_whitebox_without_container_xgb():
    from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark as Benchmark
    b = Benchmark(task_id=167199, rng=0)
    cs = b.get_configuration_space(seed=0)

    configuration = cs.get_default_configuration()
    assert configuration['colsample_bylevel'] == 1.0
    assert len(configuration.keys()) == 8

    n_estimator = 32
    subsample = 1
    result_dict = b.objective_function(configuration, fidelity=dict(n_estimators=n_estimator, dataset_fraction=subsample),
                                       rng=0)
    valid_loss = result_dict['function_value']
    train_loss = result_dict['info']['train_loss']

    result_dict = b.objective_function_test(configuration, fidelity=dict(n_estimators=n_estimator), rng=0)
    test_loss = result_dict['function_value']

    assert np.isclose(train_loss, 0.02678, atol=0.001)
    assert np.isclose(valid_loss, 0.49549, atol=0.001)
    assert np.isclose(test_loss, 0.43636, atol=0.001)


@pytest.mark.skipif(skip_container_test, reason="Requires singularity and flask")
def test_whitebox_with_container():
    from hpobench.container.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark as Benchmark
    b = Benchmark(container_name='xgboost_benchmark',
                  task_id=167199,
                  rng=0)

    cs = b.get_configuration_space()
    configuration = cs.get_default_configuration()
    assert configuration['colsample_bylevel'] == 1.0
    assert len(configuration.keys()) == 8

    n_estimator = 32
    subsample = 1
    result_dict = b.objective_function(configuration, fidelity=dict(n_estimators=n_estimator,
                                                                    dataset_fraction=subsample))
    valid_loss = result_dict['function_value']
    train_loss = result_dict['info']['train_loss']
    result_dict = b.objective_function_test(configuration, fidelity=dict(n_estimators=n_estimator))
    test_loss = result_dict['function_value']

    assert np.isclose(train_loss, 0.02232, atol=0.001)
    assert np.isclose(valid_loss, 0.4234, atol=0.001)
    assert np.isclose(test_loss, 0.43636, atol=0.001)


def test_cartpole():
    from hpobench.container.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
    b = Benchmark(container_name='cartpole',
                  rng=1)
    cs = b.get_configuration_space(seed=1)
    print(cs.get_default_configuration())

    from hpobench.container.benchmarks.rl.cartpole import CartpoleFull as Benchmark
    b = Benchmark(container_name='cartpole',
                  rng=1)
    cs = b.get_configuration_space(seed=1)
    print(cs.get_default_configuration())
