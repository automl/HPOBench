import importlib
import logging
import os

from hpobench import config


def set_log_level(debug):
    os.environ['HPOBENCH_DEBUG'] = 'true' if debug else 'false'
    import hpobench.container.client_abstract_benchmark as client
    importlib.reload(client)


def test_debug_env_variable_1():
    set_log_level(False)
    from hpobench.container.client_abstract_benchmark import LOG_LEVEL
    assert LOG_LEVEL == logging.INFO

    set_log_level(True)
    from hpobench.container.client_abstract_benchmark import LOG_LEVEL
    assert LOG_LEVEL == logging.DEBUG


def test_debug_container():
    # Test if the debug option works. Check if some debug output from the server is visible.

    set_log_level(True)

    from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    from hpobench.util.openml_data_manager import get_openmlcc18_taskids

    task_id = get_openmlcc18_taskids()[0]

    b = Benchmark(
        task_id=task_id,
        container_name='ml_mmfb',
        container_source=config.config_file.container_source,
    )

    cs = b.get_configuration_space()
    assert cs is not None

    set_log_level(False)
