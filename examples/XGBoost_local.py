import logging
from time import time

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

b = Benchmark(task_id=167149)
print(b.get_meta_information())
start = time()

values = []
cs = b.get_configuration_space()

for i in range(5):
    for n_estimator in [2, 4, 8, 16, 32]:
        for subsample in [0.1, 0.2, 0.4, 0.8, 1]:
            configuration = cs.sample_configuration()
            result_dict = b.objective_function(configuration, n_estimators=n_estimator, subsample=subsample)
            loss = result_dict['function_value']
            print(f'[{i+1}|1000] No Estimator: {n_estimator} - Subsample Rate: {subsample} - Loss {loss:.4f}')
            values.append(loss)

print(f'Done, took totally {time()-start:.2f}')
