import logging
import time
import numpy as np

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark

myrng = np.random.RandomState(10)

# container_name must be the exact same as the suffix in the recipe name
# (Singuarity.XGBoostBenchmark --> XGBoostBenchmark)
b = Benchmark(rng=myrng,
              container_name='XGBoostBenchmark',
              task_id=167149)
print(b.get_meta_information())

start = time.time()
values = []
cs = b.get_configuration_space()

num_epochs = 10
for i in range(num_epochs):
    configuration = cs.get_default_configuration()
    for n_estimator in [2, 4, 8, 16, 32]:
        for subsample in [0.1, 0.2, 0.4, 0.8, 1]:
            result_dict = b.objective_function(configuration, n_estimators=n_estimator, subsample=subsample)
            valid_loss = result_dict['function_value']
            train_loss = result_dict['train_loss']
            result_dict = b.objective_function_test(configuration, n_estimators=n_estimator)
            test_loss = result_dict['function_value']
            print(f'[{i+1}|{num_epochs}] No Estimator: {n_estimator:3d} - Subsample Rate: {subsample:.1f} - Test {test_loss:.4f} '
                  f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')
            values.append(test_loss)
        print('\n')

print("Done, took totally %.2f s" % (time.time() - start))
