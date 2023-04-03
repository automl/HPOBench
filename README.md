# HPOBench

HPOBench is a library for providing benchmarks for (multi-fidelity) hyperparameter optimization and with a focus on reproducibility.

Further info:
  * list of [benchmarks](https://github.com/automl/HPOBench/wiki/Available-Containerized-Benchmarks)
  * [howto](https://github.com/automl/HPOBench/wiki/How-to-add-a-new-benchmark-step-by-step) contribute benchmarks

## Status

Status for Master Branch: 
[![Build Status](https://github.com/automl/HPOBench/workflows/Test%20Pull%20Requests/badge.svg?branch=master)](https://github.com/automl/HPOBench/actions)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)

Status for Development Branch: 
[![Build Status](https://github.com/automl/HPOBench/workflows/Test%20Pull%20Requests/badge.svg?branch=development)](https://github.com/automl/HPOBench/actions)
[![codecov](https://codecov.io/gh/automl/HPOBench/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/HPOBench)

## In 4 lines of code

Evaluate a random configuration using a singularity container
```python
from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
b = SliceLocalizationBenchmark(rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"budget": 100}, rng=1)
```

All benchmarks can also be queried with fewer or no fidelities:

```python
from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
b = SliceLocalizationBenchmark(rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, fidelity={"budget": 50}, rng=1)
# returns results on the highest budget
result_dict = b.objective_function(configuration=config, rng=1)
```

For each benchmark further info on the searchspace and fidelity space can be obtained:

```python
from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark
b = SliceLocalizationBenchmark(task_id=167149, rng=1)
cs = b.get_configuration_space(seed=1)
fs = b.get_fidelity_space(seed=1)
meta = b.get_meta_information()
```

## Installation

We recommend using a virtual environment. To install HPOBench, please run the following:
```
git clone https://github.com/automl/HPOBench.git
cd HPOBench 
pip install .
```

**Note:** This does not install *singularity (version 3.6)*. Please follow the steps described here: [user-guide](https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps).   
If you run into problems, using the most recent singularity version might help: [here](https://singularity.hpcng.org/admin-docs/master/installation.html)

## Containerized Benchmarks

We provide all benchmarks as containerized versions to (i) isolate their dependencies and (ii) keep them reproducible. Our containerized benchmarks do not rely on external dependencies and thus do not change over time. For this, we rely on [Singularity (version 3.6)](https://sylabs.io/guides/3.6/user-guide/) and for now upload all containers to a [gitlab registry](https://gitlab.tf.uni-freiburg.de/muelleph/hpobench-registry/container_registry)

The only other requirements are: [ConfigSpace](https://github.com/automl/ConfigSpace), *scipy* and *numpy* 

### Run a Benchmark Locally

Each benchmark can also be run locally, but the dependencies must be installed manually and might conflict with other benchmarks. This can be arbitrarily complex and further information can be found in the docstring of the benchmark.
 
A simple example is the XGBoost benchmark which can be installed with `pip install .[xgboost]`

```python
from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark

b = XGBoostBenchmark(task_id=167149)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config,
                                   fidelity={"n_estimators": 128, "dataset_fraction": 0.5}, rng=1)

```

### How to Build a Container Locally

With singularity installed run the following to built the, e.g. xgboost container

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

## Configure HPOBench

All of HPOBench's settings are stored in a file, the `hpobenchrc`-file. It is a .yaml file, which is automatically generated at the first use of HPOBench. 
By default, it is placed in `$XDG_CONFIG_HOME` (or if not set this defaults to `'~/.config/hpobench'`). This file defines where to store containers and datasets and much more. We highly recommend to have a look at this file once it's created. Furthermore, please make sure to have write permission in these directories or adapt if necessary. For more information on where data is stored, please see the section on `HPOBench Data` below.

Furthermore, for running containers, we rely on Unix sockets which by default are located in `$TEMP_DIR` (or if not set this defaults to `\tmp`). 

### Remove all data, containers, and caches

Feel free to use `hpobench/util/clean_up_script.py` to remove all data, downloaded containers and caches:
```bash
python ./hpobench/util/clean_up_script.py --help
``` 

If you like to delete only specific parts, i.e. a single container, you can find the benchmark's data, container, and caches in the following directories:

#### HPOBench Data
HPOBench stores downloaded containers and datasets at the following locations:

```bash
$XDG_CONFIG_HOME # ~/.config/hpobench
$XDG_CACHE_HOME # ~/.cache/hpobench
$XDG_DATA_HOME # ~/.local/share/hpobench
```

For crashes or when not properly shutting down containers, there might be socket files left under `/tmp/hpobench_socket`.

#### OpenML Data

OpenML data additionally maintains its cache which is located at `~/.openml/`

#### Singularity Containers

Singularity additionally maintains its cache which can be removed with `singularity cache clean`

### Use HPOBench Benchmarks in Research Projects

If you use a benchmark in your experiments, please specify the version number of the HPOBench as well as the version of 
the used container to ensure reproducibility. When starting an experiment, HPOBench writes automatically these two version numbers to the log. 

### Troubleshooting and Further Notes

  - **Singularity throws an 'Invalid Image format' exception**
  Use a singularity version > 3. For users of the Meta-Cluster in Freiburg, you have to set the following path:
  ```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```

  - **A Benchmark fails with `SystemError: Could not start an instance of the benchmark. Retried 5 times` but the container 
can be started locally with `singularity instance start <pathtocontainer> test`**
See whether in `~/.singularity/instances/sing/$HOSTNAME/*/` there is a file that does not end with '}'. If yes delete this file and retry.   

**Note:** If you are looking for a different or older version of our benchmarking library, you might be looking for
 [HPOlib1.5](https://github.com/automl/HPOlib1.5) 
 
## Reference

If you use HPOBench, please cite the following paper:

```bibtex
@inproceedings{
  eggensperger2021hpobench,
  title={{HPOB}ench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for {HPO}},
  author={Katharina Eggensperger and Philipp M{\"u}ller and Neeratyoy Mallik and Matthias Feurer and Rene Sass and Aaron Klein and Noor Awad and Marius Lindauer and Frank Hutter},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=1k4rJYEwda-}
}
```

