import importlib
import logging
import os


def set_log_level(debug):
    os.environ['HPOBENCH_DEBUG'] = 'true' if debug else 'false'
    import hpobench.container.client_abstract_benchmark as client
    importlib.reload(client)


def test_debug_env_variable_1():
    set_log_level(False)
    from hpobench.container.client_abstract_benchmark import log_level
    assert log_level == logging.INFO

    set_log_level(True)
    from hpobench.container.client_abstract_benchmark import log_level
    assert log_level == logging.DEBUG


def test_debug_container():
    # Test if the debug option works. Check if some debug output from the server is visible.

    set_log_level(True)

    from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
    from hpobench.util.openml_data_manager import get_openmlcc18_taskids

    task_id = get_openmlcc18_taskids()[0]

    b = Benchmark(task_id=task_id,
                  container_name='xgboost_benchmark',
                  container_source='library://phmueller/automl')
    cs = b.get_configuration_space()
    assert cs is not None

    set_log_level(False)


def test_benchmark_encoder():
    from enum import Enum
    class test_enum(Enum):
        obj = 'name'

        def __str__(self):
            return str(self.value)

    from hpobench.container.server_abstract_benchmark import BenchmarkEncoder
    import json
    import numpy as np

    enum_obj = test_enum.obj
    enum_obj_str = json.dumps(enum_obj, cls=BenchmarkEncoder)
    assert enum_obj_str == '"name"'

    array = np.array([1, 2, 3, 4])
    array_str = json.dumps(array, cls=BenchmarkEncoder)
    assert array_str == '[1, 2, 3, 4]'


if __name__ == '__main__':
    test_debug_env_variable_1()
    test_debug_container()
