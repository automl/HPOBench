import logging
from time import time

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark

b = Benchmark(task_id=167199)
cs = b.get_configuration_space()

start = time()
for i in range(5):
    configuration = cs.sample_configuration()
    print(configuration)
    for n_estimator in [2, 4, 8, 16, 32]:
        for subsample in [0.1, 0.2, 0.4, 0.8, 1]:
            result_dict = b.objective_function(configuration, n_estimators=n_estimator, subsample=subsample)
            valid_loss = result_dict['function_value']
            train_loss = result_dict['train_loss']

            result_dict = b.objective_function_test(configuration, n_estimators=n_estimator)
            test_loss = result_dict['function_value']

            print(f'[{i+1}|5] No Estimator: {n_estimator:3d} - Subsample Rate: {subsample:.1f} - Test {test_loss:.4f} '
                  f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')

print(f'Done, took totally {time()-start:.2f}')
