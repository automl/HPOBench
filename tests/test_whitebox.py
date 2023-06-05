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
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    b = Benchmark(task_id=146818, rng=0)
    cs = b.get_configuration_space(seed=0)

    configuration = cs.get_default_configuration()
    assert configuration['colsample_bytree'] == 1.0
    assert len(configuration.keys()) == 4

    n_estimator = 100
    subsample = 1
    result_dict = b.objective_function(
        configuration, fidelity=dict(n_estimators=n_estimator, subsample=subsample), rng=0
    )
    valid_loss = result_dict['function_value']
    train_loss = result_dict['info']['train_loss']

    result_dict = b.objective_function_test(
        configuration, fidelity=dict(n_estimators=n_estimator), rng=0
    )
    test_loss = result_dict['function_value']

    assert train_loss == pytest.approx(1.0, abs=0.001)
    assert valid_loss == pytest.approx(0.166, abs=0.001)
    assert test_loss == pytest.approx(0.087, abs=0.001)


@pytest.mark.skipif(skip_container_test, reason="Requires singularity and flask")
def test_whitebox_with_container():
    from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    b = Benchmark(task_id=146818, rng=0)  #, container_name='ml_mmfb',)

    cs = b.get_configuration_space()
    configuration = cs.get_default_configuration()
    assert configuration['colsample_bytree'] == 1.0
    assert len(configuration.keys()) == 4

    n_estimator = 100
    subsample = 1
    result_dict = b.objective_function(
        configuration, fidelity=dict(n_estimators=n_estimator, subsample=subsample)
    )
    valid_loss = result_dict['function_value']
    train_loss = result_dict['info']['train_loss']
    result_dict = b.objective_function_test(configuration, fidelity=dict(n_estimators=n_estimator))
    test_loss = result_dict['function_value']
    
    assert train_loss == pytest.approx(1.0, abs=0.001)
    assert valid_loss == pytest.approx(0.1512, abs=0.001)
    assert test_loss == pytest.approx(0.1014, abs=0.001)


@pytest.mark.skipif(skip_container_test, reason="Requires singularity and flask")
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
