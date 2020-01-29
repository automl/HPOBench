import logging
from time import time

from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostOnMnist as Benchmark

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

b = Benchmark()
print(b.get_meta_information())
start = time()

values = []
cs = b.get_configuration_space()

for i in range(1000):
    configuration = cs.sample_configuration()
    rval = b.objective_function(configuration, n_estimators=5, subsample=0.1)
    loss = rval['function_value']
    print(f'[{i+1}|1000]Loss {loss:.4f}')

    values.append(loss)

print(f'Done, took totally {time()-start:.2f}')
