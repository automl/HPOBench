# HPOBench

HPOBench is a library for hyperparameter optimization and black-box optimization benchmark with a focus on reproducibility.

**Note:** HPOBench is under active construction. Stay tuned for more benchmarks. Information on how to contribute a new benchmark will follow shortly.

**Note:** If you are looking for a different version of HPOBench, you might be looking for [HPOlib1.5](https://github.com/automl/HPOlib1.5) 

## In 4 lines of code

Run a random configuration within a singularity container
```python
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149, container_source='library://phmueller/automl', rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"n_estimators": 128, "subsample": 0.5}, rng=1)
```

All benchmarks can also be queried with fewer or no fidelities:

```python
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149, container_source='library://phmueller/automl', rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"n_estimators": 128,}, rng=1)
result_dict = b.objective_function(configuration=config, rng=1)
```

Containerized benchmarks do not rely on external dependencies and thus do not change. To do so, we rely on [Singularity (version 3.5)](https://sylabs.io/guides/3.5/user-guide/).
 
Further requirements are: [ConfigSpace](https://github.com/automl/ConfigSpace), *scipy* and *numpy* 

**Note:** Each benchmark can also be run locally, but the dependencies must be installed manually and might conflict with other benchmarks. 
 This can be arbitrarily complex and further information can be found in the docstring of the benchmark.
 
A simple example is the XGBoost benchmark which can be installed with `pip install .[xgboost]`
```python
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"n_estimators": 128, "subsample": 0.5}, rng=1)

```

## Installation

Before we start, we recommend using a virtual environment. To run any benchmark using its singularity container, 
run the following:
```
git clone https://github.com/automl/HPOBench.git
cd HPOBench 
pip install .
```

**Note:** This does not install *singularity (version 3.5)*. Please follow the steps described here: [user-guide](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps).   

## Available Containerized Benchmarks

| Benchmark Name                    | Container Name     | Container Source                     | Hosted at | Additional Info                      |
| :-------------------------------- | ------------------ | ------------------------------------ | ----------|-------------------------------------- |
| XGBoostBenchmark                  | xgboost_benchmark  | library://phmueller/automl/xgboost_benchmark | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f610eae86dd3232deb5a5) | Works with OpenML task ids |
| CartpoleFull                      | cartpole           | library://phmueller/automl/cartpole  | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f310084a01836e4395601) | Not deterministic                    |
| CartpoleReduced                   | cartpole           | library://phmueller/automl/cartpole  | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f310084a01836e4395601) | Not deterministic                    |
| Learna                            | learna_benchmark   | library://phmueller/automl/learna_benchmark | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f31c3b1793638c1134e58) | Not deterministic                    |
| MetaLearna                        | learna_benchmark   | library://phmueller/automl/learna_benchmark | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f31c3b1793638c1134e58) | Not deterministic                    |
| SliceLocalizationBenchmark        | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f630cb1793638c1134e5d) | Loading may take several minutes     |
| ProteinStructureBenchmark         | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f630cb1793638c1134e5d) | Loading may take several minutes     |
| NavalPropulsionBenchmark          | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f630cb1793638c1134e5d) | Loading may take several minutes     |
| ParkinsonsTelemonitoringBenchmark | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/_container/5f0f630cb1793638c1134e5d) | Loading may take several minutes     |
| NASCifar10ABenchmark              | nasbench_101       | library://phmueller/automl/nasbench_101 | [Sylabs](https://cloud.sylabs.io/library/_container/5f227263b1793638c1135c37) |                                     |
| NASCifar10BBenchmark              | nasbench_101       | library://phmueller/automl/nasbench_101 | [Sylabs](https://cloud.sylabs.io/library/_container/5f227263b1793638c1135c37) |                                     |
| NASCifar10CBenchmark              | nasbench_101       | library://phmueller/automl/nasbench_101 | [Sylabs](https://cloud.sylabs.io/library/_container/5f227263b1793638c1135c37) |                                     |

## Further Notes

### How to build a container locally

With singularity installed run the following to built the xgboost container

```bash
cd hpobench/container/recipes/ml
sudo singularity build xgboost_benchmark Singularity.XGBoostBenchmark
```

You can use this local image with:

```python
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149, container_name="xgboost_benchmark", 
                     container_source='./') # path to hpobench/container/recipes/ml
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(config, fidelity={"n_estimators": 128, "subsample": 0.5})
```

### Remove all caches

#### HPOBench data
HPOBench stores downloaded containers and datasets at the following locations:

```bash
$XDG_CONFIG_HOME # ~/.config/hpobench
$XDG_CACHE_HOME # ~/.config/hpobench
$XDG_DATA_HOME # ~/.cache/hpobench
```

For crashes or when not properly shutting down containers, there might be socket files left under `/tmp/`.

#### OpenML data

OpenML data additionally maintains it's own cache with is found at `~/.openml/`

#### Singularity container

Singularity additionally maintains it's own cache which can be removed with `singularity cache clean`

### Troubleshooting

  - **Singularity throws an 'Invalid Image format' exception**
  Use a singularity version > 3. For users of the Meta-Cluster in Freiburg, you have to set the following path:
  ```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```

  - **A Benchmark fails with `SystemError: Could not start a instance of the benchmark. Retried 5 times` but the container 
can be started locally with `singularity instance start <pathtocontainer> test`**
See whether in `~/.singularity/instances/sing/$HOSTNAME/*/` there is a file that does not end with '}'. If yes delete this file and retry.   
  
## Status

Status for Master Branch: 

[![Build Status](https://travis-ci.org/automl/HPOBench.svg?branch=master)](https://travis-ci.org/automl/HPOBench)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)

Status for Development Branch: 

[![Build Status](https://travis-ci.org/automl/HPOBench.svg?branch=development)](https://travis-ci.org/automl/HPOBench)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)
