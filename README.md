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
from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark

b = XGBoostBenchmark(task_id=167149)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config,
                                   fidelity={"n_estimators": 128, "dataset_fraction": 0.5}, rng=1)

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

| Benchmark Name                    | Container Name     | Additional Info                      |
| :-------------------------------- | ------------------ | ------------------------------------ |
| BNNOn*                            | pybnn              | There are 4 benchmark in total (ToyFunction, BostonHousing, ProteinStructure, YearPrediction) |
| CartpoleFull                      | cartpole           | Not deterministic.                    |
| CartpoleReduced                   | cartpole           | Not deterministic.                    |
| SliceLocalizationBenchmark        | tabular_benchmarks | Loading may take several minutes.     |
| ProteinStructureBenchmark         | tabular_benchmarks | Loading may take several minutes.     |
| NavalPropulsionBenchmark          | tabular_benchmarks | Loading may take several minutes.     |
| ParkinsonsTelemonitoringBenchmark | tabular_benchmarks | Loading may take several minutes.     |
| NASCifar10*Benchmark              | nasbench_101       | Loading may take several minutes. There are 3 benchmark in total (A, B, C) |
| *NasBench201Benchmark             | nasbench_201       | Loading may take several minutes. There are 3 benchmarks in total (Cifar10Valid, Cifar100, ImageNet)    |
| NASBench1shot1SearchSpace*Benchmark | nasbench_1shot1  | Loading may take several minutes. There are 3 benchmarks in total (1,2,3) |
| ParamNet*OnStepsBenchmark       | paramnet         | There are 6 benchmarks in total (Adult, Higgs, Letter, Mnist, Optdigits, Poker) |
| ParamNet*OnTimeBenchmark        | paramnet         | There are 6 benchmarks in total (Adult, Higgs, Letter, Mnist, Optdigits, Poker) |
| SurrogateSVMBenchmark              | surrogate_svm      | Random Forest Surrogate of a SVM on MNIST | 
| Learna⁺                            | learna_benchmark   | Not deterministic.                    |
| MetaLearna⁺                        | learna_benchmark   | Not deterministic.                    |
| XGBoostBenchmark⁺                  | xgboost_benchmark  | Works with OpenML task ids. |
| XGBoostExtendedBenchmark⁺          | xgboost_benchmark  | Works with OpenML task ids + Contains Additional Parameter `Booster |
| SupportVectorMachine⁺              | svm_benchmark      | Works with OpenML task ids. |

⁺ these benchmarks are not yet final and might change

**Note:** All containers are uploaded [here](https://gitlab.tf.uni-freiburg.de/muelleph/hpobench-registry/container_registry)

## Further Notes

### Configure the HPOBench

All of HPOBench's settings are stored in a file, the `hpobenchrc`-file. 
It is a yaml file, which is automatically generated at the first use of HPOBench. 
By default, it is placed in `$XDG_CONFIG_HOME`. If `$XDG_CONFIG_HOME` is not set, then the
`hpobenchrc`-file is saved to `'~/.config/hpobench'`. When using the containerized benchmarks, the Unix socket is 
defined via `$TEMP_DIR`. This is by default `\tmp`. Make sure to have write permissions in those directories. 

In the `hpobenchrc`, you can specify for example the directory, in that the benchmark containers are
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

### Remove all data, containers, and caches

Update: In version 0.0.8, we have added the script `hpobench/util/clean_up_script.py`. It allows to easily remove all
data, downloaded containers, and caches. To get more information, you can use the following command. 
```bash
python ./hpobench/util/clean_up_script.py --help
``` 

If you like to delete only specific parts, i.e. a single container,
you can find the benchmark's data, container, and caches in the following directories:

#### HPOBench data
HPOBench stores downloaded containers and datasets at the following locations:

```bash
$XDG_CONFIG_HOME # ~/.config/hpobench
$XDG_CACHE_HOME # ~/.cache/hpobench
$XDG_DATA_HOME # ~/.local/share/hpobench
```

For crashes or when not properly shutting down containers, there might be socket files left under `/tmp/hpobench_socket`.

#### OpenML data

OpenML data additionally maintains its cache which is located at `~/.openml/`

#### Singularity container

Singularity additionally maintains its cache which can be removed with `singularity cache clean`

### Use HPOBench benchmarks in research projects

If you use a benchmark in your experiments, please specify the version number of the HPOBench as well as the version of 
the used container. When starting an experiment, HPOBench writes automatically the 2 version numbers to the log. 

### Troubleshooting

  - **Singularity throws an 'Invalid Image format' exception**
  Use a singularity version > 3. For users of the Meta-Cluster in Freiburg, you have to set the following path:
  ```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```

  - **A Benchmark fails with `SystemError: Could not start an instance of the benchmark. Retried 5 times` but the container 
can be started locally with `singularity instance start <pathtocontainer> test`**
See whether in `~/.singularity/instances/sing/$HOSTNAME/*/` there is a file that does not end with '}'. If yes delete this file and retry.   
  
## Status

Status for Master Branch: 
[![Build Status](https://github.com/automl/HPOBench/workflows/Test%20Pull%20Requests/badge.svg?branch=master)](https://https://github.com/automl/HPOBench/actions)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)

Status for Development Branch: 
[![Build Status](https://github.com/automl/HPOBench/workflows/Test%20Pull%20Requests/badge.svg?branch=development)](https://https://github.com/automl/HPOBench/actions)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)
