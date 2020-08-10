import pytest
import os
import logging
import importlib


def test_debug_env_variable_1():
    os.environ['HPOLIB_DEBUG'] = 'false'
    from hpolib.container.client_abstract_benchmark import log_level
    assert log_level == logging.INFO

    os.environ['HPOLIB_DEBUG'] = 'true'
    import hpolib.container.client_abstract_benchmark as client
    importlib.reload(client)
    from hpolib.container.client_abstract_benchmark import log_level
    assert log_level == logging.DEBUG


def test_debug_container():
    # Test if the debug option works. Check if some debug output from the server is visible.

    os.environ['HPOLIB_DEBUG'] = 'true'

    import hpolib.container.client_abstract_benchmark as client
    importlib.reload(client)

    from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    from hpolib.util.openml_data_manager import get_openmlcc18_taskids

    task_id = get_openmlcc18_taskids()[0]

    b = Benchmark(task_id=task_id,
                  container_name='xgboost_benchmark',
                  container_source='library://phmueller/automl')
    cs = b.get_configuration_space()
    assert cs is not None


if __name__ == '__main__':
    test_debug_env_variable_1()
    test_debug_container()
