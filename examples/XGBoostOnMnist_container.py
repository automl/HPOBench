import logging
logging.basicConfig(level=logging.DEBUG)
import time
import numpy as np

from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostOnMnist as Benchmark

myrng = np.abs(10)
myrng = np.random.RandomState(myrng)

# imgName must be the exact same as the suffix in the recipe name (Singuarity.XGBoostOnMnist)
b = Benchmark(rng=myrng, imgName='XGBoostOnMnist')
print(b.get_meta_information())

start = time.time()
values = []
cs = b.get_configuration_space()

for i in range(1000):
    configuration = cs.sample_configuration()
    rval = b.objective_function(configuration, n_estimators=5, subsample=0.1)
    loss = rval['function_value']
    print(f'[{i+1}|1000]Loss {loss:.4f}')

    values.append(loss)

print("Done, took totally %.2f s" % ((time.time() - start)))
