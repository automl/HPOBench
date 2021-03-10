# HPOBench

HPOBench is a library for hyperparameter optimization and black-box optimization benchmark with a focus on reproducibility.

**Note:** HPOBench is under active construction. Stay tuned for more benchmarks. Information on how to contribute a new benchmark will follow shortly.

**Note:** If you are looking for a different or older version of our benchmarking library, you might be looking for
 [HPOlib1.5](https://github.com/automl/HPOlib1.5) 

## In 4 lines of code

Run a random configuration within a singularity container
```python
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
b = XGBoostBenchmark(task_id=167149, container_source='library://phmueller/automl', rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"n_estimators": 128, "dataset_fraction": 0.5}, rng=1)
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
result_dict = b.objective_function(configuration=config, fidelity={"n_estimators": 128, "dataset_fraction": 0.5}, rng=1)

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
| XGBoostBenchmark                  | xgboost_benchmark  | library://phmueller/automl/xgboost_benchmark | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Works with OpenML task ids |
| XGBoostExtendedBenchmark           | xgboost_benchmark  | library://phmueller/automl/xgboost_benchmark | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Works with OpenML task ids + Contains Additional Parameter `Booster |
| SupportVectorMachine              | svm_benchmark      | library://phmueller/automl/svm_benchmark | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Works with OpenML task ids |
| BNNOnToyFunction                  | pybnn              | library://phmueller/automl/pybnn     | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |  |
| BNNOnBostonHousing                | pybnn              | library://phmueller/automl/pybnn     | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |  |
| BNNOnProteinStructure             | pybnn              | library://phmueller/automl/pybnn     | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |  |
| BNNOnYearPrediction               | pybnn              | library://phmueller/automl/pybnn     | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |  |
| CartpoleFull                      | cartpole           | library://phmueller/automl/cartpole  | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Not deterministic                    |
| CartpoleReduced                   | cartpole           | library://phmueller/automl/cartpole  | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Not deterministic                    |
| Learna                            | learna_benchmark   | library://phmueller/automl/learna_benchmark | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Not deterministic                    |
| MetaLearna                        | learna_benchmark   | library://phmueller/automl/learna_benchmark | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Not deterministic                    |
| SliceLocalizationBenchmark        | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| ProteinStructureBenchmark         | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| NavalPropulsionBenchmark          | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| ParkinsonsTelemonitoringBenchmark | tabular_benchmarks | library://phmueller/automl/tabular_benchmarks | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| NASCifar10ABenchmark              | nasbench_101       | library://phmueller/automl/nasbench_101 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| NASCifar10BBenchmark              | nasbench_101       | library://phmueller/automl/nasbench_101 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| NASCifar10CBenchmark              | nasbench_101       | library://phmueller/automl/nasbench_101 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes     |
| Cifar10NasBench201Benchmark       | nasbench_201       | library://phmueller/automl/nasbench_201 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| Cifar100NasBench201Benchmark      | nasbench_201       | library://phmueller/automl/nasbench_201 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| Cifar10ValidNasBench201Benchmark  | nasbench_201       | library://phmueller/automl/nasbench_201 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| ImageNetNasBench201Benchmark      | nasbench_201       | library://phmueller/automl/nasbench_201 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| NASBench1shot1SearchSpace1Benchmark | nasbench_1shot1  | library://phmueller/automl/nasbench_1shot1 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| NASBench1shot1SearchSpace2Benchmark | nasbench_1shot1  | library://phmueller/automl/nasbench_1shot1 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| NASBench1shot1SearchSpace3Benchmark | nasbench_1shot1  | library://phmueller/automl/nasbench_1shot1 | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) | Loading may take several minutes    |
| ParamNetAdultOnStepsBenchmark       | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetAdultOnTimeBenchmark        | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetHiggsOnStepsBenchmark       | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetHiggsOnTimeBenchmark        | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetLetterOnStepsBenchmark      | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetLetterOnTimeBenchmark       | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetMnistOnStepsBenchmark       | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetMnistOnTimeBenchmark        | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetOptdigitsOnStepsBenchmark   | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetOptdigitsOnTimeBenchmark    | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetPokerOnStepsBenchmark       | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetPokerOnTimeBenchmark        | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetVehicleOnStepsBenchmark     | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |
| ParamNetVehicleOnTimeBenchmark      | paramnet         | library://phmueller/automl/paramnet | [Sylabs](https://cloud.sylabs.io/library/phmueller/automl) |     |

## Further Notes

### Configure the HPOBench

All of HPOBench's settings are stored in a file, the `hpobenchrc`-file. 
It is a yaml file, which is automatically generated at the first use of HPOBench. 
By default, it is placed in `$XDG_CONFIG_HOME`. If `$XDG_CONFIG_HOME` is not set, then the
`hpobenchrc`-file is saved to `'~/.config/hpobench'`.
Make sure to have write permissions in this directory. 

In the `hpobenchrc`, you can specify for example the directory, in that the benchmark-containers are
downloaded. We encourage you to take a look into the `hpobenchrc`, to find out more about all
possible settings. 


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
result_dict = b.objective_function(config, fidelity={"n_estimators": 128, "dataset_fraction": 0.5})
```

### Remove all caches

#### HPOBench data
HPOBench stores downloaded containers and datasets at the following locations:

```bash
$XDG_CONFIG_HOME # ~/.config/hpobench
$XDG_CACHE_HOME # ~/.cache/hpobench
$XDG_DATA_HOME # ~/.local/share/hpobench
```

For crashes or when not properly shutting down containers, there might be socket files left under `/tmp/`.

#### OpenML data

OpenML data additionally maintains it's own cache which is located at `~/.openml/`

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
[![Build Status](https://github.com/automl/HPOBench/workflows/Test%20Pull%20Requests/badge.svg?branch=master)](https://https://github.com/automl/HPOBench/actions)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)

Status for Development Branch: 
[![Build Status](https://github.com/automl/HPOBench/workflows/Test%20Pull%20Requests/badge.svg?branch=development)](https://https://github.com/automl/HPOBench/actions)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)
