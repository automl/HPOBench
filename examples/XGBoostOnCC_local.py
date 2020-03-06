import logging
from time import time

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
from hpolib.util.openml_data_manager import get_openmlcc18_taskids

logger = logging.getLogger('XGBoostBenchmark')
logger.setLevel(level=logging.DEBUG)

task_ids = get_openmlcc18_taskids()

start = time()

for task_no, task_id in enumerate(task_ids):
    logger.info(f'###################### TASK {task_no + 1} of {len(task_ids)} ######################')
    b = Benchmark(task_id=task_id)
    cs = b.get_configuration_space()
    for i in range(3):
        config = cs.sample_configuration()
        rval = b.objective_function(config, n_estimators=5, subsample=0.1)
        logger.info(f'[{task_id}][{i + 1}|3]Loss {rval.get("function_value"):.4f}')

print(f'Done, took totally {time() - start:.2f}')
