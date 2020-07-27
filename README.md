# HPOlib3

HPOlib3 is a library for hyperparameter optimization and black-box optimization benchmark with a focus on reproducibility.

**Note:** Hpolib3 is under active construction. Stay tuned for more benchmarks. Information on how to contribution a new benchmark will follow shortly.

# HPOlib3 in 4 lines of code

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

HPOlib3 is built such that it prevents the benchmark's behavior to change due to updates on required packages, we host for each benchmark a containerized version. 
For this purpose, we rely on [Singularity (version 3.5)](https://sylabs.io/guides/3.5/user-guide/). To install singularity, 
please follow the instructions in its [user-guide](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps).   
However, each benchmark can also be used without singularity, but the dependencies might conflict.
 
Further requirements are: the [ConfigSpace](https://github.com/automl/ConfigSpace) package, *scikit-learn*, *scipy*, *numpy*, 
*python-openml*. 

## Installation
We explain in the docstrings of each benchmark, how to install the dependencies for each benchmark properly. \
For the purpose of this documentation, we show exemplarily how to install everything for the xgboost benchmark. 
Before we start, we recommend using a virtual environment.
 
In general, we can install HPOLib3 and specify its extra requirements via \
``` pip install <path_to_hpolib3_repository>[extra requirements] ```\

For the Xgboost benchmark we need the following command:
``` pip install .[xgboost,singularity] ```
This installs xgboost as well as the requirements for singularity in your python environment. 
(Note: It doesn't install singularity. Please install it using the link above.) 

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
| <font size="2em">Cifar10NasBench201Benchmark</font>       | <font size="2em">nasbench_201</font>       | <font size="2em">library://phmueller/automl/nasbench_201</font> | <font size="2em"> </font>                                    |
| <font size="2em">Cifar10ValidNasBench201Benchmark</font>  | <font size="2em">nasbench_201</font>       | <font size="2em">library://phmueller/automl/nasbench_201</font> | <font size="2em"> </font>                                    |
| <font size="2em">Cifar100NasBench201Benchmark</font>      | <font size="2em">nasbench_201</font>       | <font size="2em">library://phmueller/automl/nasbench_201</font> | <font size="2em"> </font>                                    |
| <font size="2em">ImageNetNasBench201Benchmark</font>      | <font size="2em">nasbench_201</font>       | <font size="2em">library://phmueller/automl/nasbench_201</font> | <font size="2em"> </font>                                    |


## Use singularity on Cluster:
For users from the university of freiburg with access to computational cluster: \\
To use the the singularity version 3.5, first you have 
to set the following path:\
```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```} \
**Note:** This works currently only on 'kisbat3'. 

## Notes: 
- The usage of different task ids is shown in the example 'XGBoost_with_container.py'
- To use a local image, (without downloading it from the sylabs-library), add the parameter 
`container-source=<path-to-directory-in-which-the-image-is>` in the Benchmark initialization.
E.g. (see XGBoost_with_container.py) \
```
b = Benchmark(rng=my_rng, container_name='xgboost_benchmark', 
              container_source=<PATH>, task_id=task_id)
```
- Singularity will throw an exception 'Invalid Image format' if you use a singularity version < 3.
  This happens, if you haven't exported the path to singularity3.5 on kisbat3 (see above).

## Status

Status for Master Branch: 

[![Build Status](https://travis-ci.org/automl/HPOlib3.svg?branch=master)](https://travis-ci.org/automl/HPOlib3)
[![codecov](https://codecov.io/gh/automl/HPOlib3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HPOlib3)

Status for Development Branch: 

[![Build Status](https://travis-ci.org/automl/HPOlib3.svg?branch=development)](https://travis-ci.org/automl/HPOlib3)
[![codecov](https://codecov.io/gh/automl/HPOlib3/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/HPOlib3)
