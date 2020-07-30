# HPOlib3

HPOlib3 is a library for hyperparameter optimization and black-box optimization benchmark with a focus on reproducibility.

**Note:** Hpolib3 is under active construction. Stay tuned for more benchmarks. Information on how to contribute a new benchmark will follow shortly.

## In 4 lines of code

Evaluate a random configuration locally (requires all dependencies for XGB to be installed)

```python
from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function_test(config, n_estimators=128, subsample=0.5)
```

Run a random configuration within a singularity container (does not need any dependencies)
```python
from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149, container_source='library://phmueller/automl')
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function_test(config, n_estimators=128, subsample=0.5)
```

Containerized benchmarks do not rely on external dependencies and thus do not change. To do so, we rely on [Singularity (version 3.5)](https://sylabs.io/guides/3.5/user-guide/). To install singularity, 
please follow the instructions in its [user-guide](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps).   
However, each benchmark can also be used without singularity, but the dependencies might conflict.
 
Further requirements are: the [ConfigSpace](https://github.com/automl/ConfigSpace) package, *scikit-learn*, *scipy*, *numpy*, 
*python-openml*. 

## Installation

Before we start, we recommend using a virtual environment.
 
In general, we can install HPOLib3 and specify its extra requirements via \
``` pip install <path_to_hpolib3_repository>[extra requirements] ```\

For the Xgboost benchmark we need the following command:
``` pip install .[singularity] ```
*Note:* This doesn't install singularity. Please install it using the link above. 
*Note:* To run a benchmark locally all dependencies for this benchmarks need to be installed. To do so, please use `pip install .[<name>]` and check 
the docstring of the benchmark you want to run locally. 

## Available Experiments with container

| Benchmark Name                                            | Container Name                             | Container Source                                             | Additional Info                                              |
| :-------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <font size="2em">XGBoostBenchmark</font>                  | <font size="2em">xgboost_benchmark</font>  | <font size="2em">library://phmueller/automl/xgboost_benchmark</font> | <font size="2em">Pass a openml task id to the dataset </font> |
| <font size="2em">CartpoleFull</font>                      | <font size="2em">cartpole</font>           | <font size="2em">library://phmueller/automl/cartpole</font>  | <font size="2em">Not deterministic</font>                    |
| <font size="2em">CartpoleReduced</font>                   | <font size="2em">cartpole</font>           | <font size="2em">library://phmueller/automl/cartpole</font>  | <font size="2em">Not deterministic</font>                    |
| <font size="2em">Learna</font>                            | <font size="2em">learna_benchmark</font>   | <font size="2em">library://phmueller/automl/learna_benchmark</font> | <font size="2em">Not deterministic</font>                    |
| <font size="2em">MetaLearna</font>                        | <font size="2em">learna_benchmark</font>   | <font size="2em">library://phmueller/automl/learna_benchmark</font> | <font size="2em">Not deterministic</font>                    |
| <font size="2em">SliceLocalizationBenchmark</font>        | <font size="2em">tabular_benchmarks</font> | <font size="2em">library://phmueller/automl/tabular_benchmarks</font> | <font size="2em">Loading may take several minutes</font>     |
| <font size="2em">ProteinStructureBenchmark</font>         | <font size="2em">tabular_benchmarks</font> | <font size="2em">library://phmueller/automl/tabular_benchmarks</font> | <font size="2em">Loading may take several minutes</font>     |
| <font size="2em">NavalPropulsionBenchmark</font>          | <font size="2em">tabular_benchmarks</font> | <font size="2em">library://phmueller/automl/tabular_benchmarks</font> | <font size="2em">Loading may take several minutes</font>     |
| <font size="2em">ParkinsonsTelemonitoringBenchmark</font> | <font size="2em">tabular_benchmarks</font> | <font size="2em">library://phmueller/automl/tabular_benchmarks</font> | <font size="2em">Loading may take several minutes</font>     |
| <font size="2em">NASCifar10ABenchmark</font>              | <font size="2em">nasbench_101</font>       | <font size="2em">library://phmueller/automl/nasbench_101</font> | <font size="2em"> </font>                                    |
| <font size="2em">NASCifar10BBenchmark</font>              | <font size="2em">nasbench_101</font>       | <font size="2em">library://phmueller/automl/nasbench_101</font> | <font size="2em"> </font>                                    |
| <font size="2em">NASCifar10CBenchmark</font>              | <font size="2em">nasbench_101</font>       | <font size="2em">library://phmueller/automl/nasbench_101</font> | <font size="2em"> </font>                                    |

## Further Notes

- The usage of different OpenML task ids is shown in the example 'XGBoost_with_container.py'
- To use a local image, (without downloading it from the sylabs-library), add the parameter 
`container-source=<path-to-directory-in-which-the-image-is>` in the Benchmark initialization.
E.g. (see XGBoost_with_container.py) \
```
b = Benchmark(rng=my_rng, container_name='xgboost_benchmark', 
              container_source=<PATH>, task_id=task_id)
```
- For users of the Meta-Cluster in Freiburg, you have to set the following path:\
```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```} \
- Singularity will throw an exception 'Invalid Image format' if you use a singularity version < 3
  
## Status

Status for Master Branch: 

[![Build Status](https://travis-ci.org/automl/HPOlib3.svg?branch=master)](https://travis-ci.org/automl/HPOlib3)
[![codecov](https://codecov.io/gh/automl/HPOlib3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HPOlib3)

Status for Development Branch: 

[![Build Status](https://travis-ci.org/automl/HPOlib3.svg?branch=development)](https://travis-ci.org/automl/HPOlib3)
[![codecov](https://codecov.io/gh/automl/HPOlib3/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/HPOlib3)
