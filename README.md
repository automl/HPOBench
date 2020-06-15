# HPOlib3
[![Build Status](https://travis-ci.org/automl/HPOlib3.svg?branch=master)](https://travis-ci.org/automl/HPOlib3) [![Build Status](https://travis-ci.org/automl/HPOlib3.svg?branch=development)](https://travis-ci.org/automl/HPOlib3)

HPOlib3 is a benchmark suite for testing hyperparameter optimizer in a easy way. HPOlib3 offers benchmarks with 
multi-fidelities.
Since we are still under heavy development, there are currently only 2 benchmarks available. A benchmark on xgboost and one on 
learning a reinforcement agent on cartpole. More benchmarks will follow soon. 

HPOlib3 implements a simple interface to interact with the benchmarks. As a configuration space for the benchmarks, 
we use our [ConfigSpace](https://github.com/automl/ConfigSpace) package. Since not every Optimizer uses our ConfigSpace, 
HPOlib3 supports calling its objective function with dictionaries as well as arrays (experimental). 
Therefore, simple transformation from different Configuration Space definitions are possible, if the hyperaparameters in 
the used Configuration Space as well as the Benchmark's internal ConfigSpace have the same names and data types. 

To prevent the benchmark's behavior to change due to updates on required packages, we host for each benchmark a containerized version. 
For this purpose, we use [Singularity (version 3.5)](https://sylabs.io/guides/3.5/user-guide/). To install singularity, 
please follow the instructions in its [user-guide](https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps).   

In HPOlib3, it can be easily switched between the usage of the containerized and local version of a benchmark.

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

#### Example on local machine

To run the XGBoost Example: \
```python3 ./HPOlib3/examples/XGBoost_local.py```

#### Example with container

To run the Example: \
```python3 ./HPOlib3/examples/XGBoost_with_container.py```

#### Use singularity on Cluster:
For users from the university of freiburg with access to computational cluster: \\
To use the the singularity version 3.5, first you have 
to set the following path:\
```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```} \
**Note:** This works currently only on 'kisbat3'. 

#### Notes: 
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

