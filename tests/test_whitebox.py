import numpy as np
import pytest
from time import time

import logging
logging.basicConfig(level=logging.DEBUG)

try:
    import Pyro4
    skip_container_test = False
except ImportError:
    skip_container_test = True


def test_whitebox_without_container():
    from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    b = Benchmark(task_id=167199, rng=0)
    cs = b.get_configuration_space(seed=0)

    start = time()
    configuration = cs.get_default_configuration()
    assert configuration['colsample_bylevel'] == 1.0
    assert len(configuration.keys()) == 6

    n_estimator = 32
    subsample = 1
    result_dict = b.objective_function(configuration, n_estimators=n_estimator, subsample=subsample, rng=0)
    valid_loss = result_dict['function_value']
    train_loss = result_dict['train_loss']

    result_dict = b.objective_function_test(configuration, n_estimators=n_estimator, rng=0)
    test_loss = result_dict['function_value']

    assert np.isclose(train_loss, 0.1071, atol=0.001)
    assert np.isclose(valid_loss, 0.3873, atol=0.001)
    assert np.isclose(test_loss, 0.38181, atol=0.001)


@pytest.mark.skipif(skip_container_test, reason="Requires singularity and flask")
def test_whitebox_with_container():
    from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    b = Benchmark(container_source='library://keggensperger/automl/',
                  container_name='xgboost_benchmark',
                  task_id=167199,
                  rng=0)

    cs = b.get_configuration_space()
    configuration = cs.get_default_configuration()
    assert configuration['colsample_bylevel'] == 1.0
    assert len(configuration.keys()) == 6

    n_estimator = 32
    subsample = 1
    result_dict = b.objective_function(configuration, n_estimators=n_estimator, subsample=subsample)
    valid_loss = result_dict['function_value']
    train_loss = result_dict['train_loss']
    result_dict = b.objective_function_test(configuration, n_estimators=n_estimator)
    test_loss = result_dict['function_value']

    print(train_loss, valid_loss, test_loss)
    assert np.isclose(train_loss, 0.1071, atol=0.001)
    assert np.isclose(valid_loss, 0.3873, atol=0.001)
    assert np.isclose(test_loss, 0.38181, atol=0.001)

