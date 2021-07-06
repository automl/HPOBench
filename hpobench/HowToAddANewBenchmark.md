# How to add a new benchmark step-by-step

## Placeholders for your benchmark

- `<type>`: Category of the benchmark, e.g. od (outlier detection)
- `<Type>`: Category of the benchmark in uppercase, e.g. OD (outlier detection)
- `<dataset_name>`: Name of the dataset (optional, ), e.g. cifar10
- `<DatasetName>`: Name of the dataset (optional), e.g. Cifar10
- `<new_benchmark>*`: Filename of the benchmark, e.g. `<type>`_ocsvm_`<dataset_name>`_benchmark or `<type>`_ocsvm_benchmark
- `<NewBenchmark>*`: Classname of the benchmark, e.g. `<Type>`OCSVM`<DatasetName>`Benchmark or `<Type>`OCSVMBenchmark
- `<branch_name>`: Branch name for your benchmarks, e.g. outlier_detection

`*`: has to be unique across all available benchmarks.


## Create a local benchmark

Clone HPOBench, switch to the development branch and create your own branch. Then install hpobench
with `pip install .`
```bash
git clone https://github.com/automl/HPOBench.git
cd HPOBench
git checkout development
git branch <branch_name>
git checkout <branch_name>
pip install .
```

Then: 
  1. Implement your new benchmark class `<NewBenchmark>` in `hpobench/benchmarks/<type>/<new_benchmark>.py` inheriting from the base class 
  `AbstractBenchmark` in `hpobench.abstract_benchmark`. Your benchmark should implement `__init__()`, 
  `get_configuration_space()`, `get_fidelity_space()`, `objective_function()` and `objective_function_test()`.
    A good example for this can be found in `hpobench/benchmarks/ml/xgboost_benchmark.py`
  3. If your benchmarks needs a dataset (e.g. for training a ml model), please also implement a DataManager, see e.g.
   `hpobench/util/openml_data_manager.py` with a `load()` method that downloads data once and reuses it for further calls.
  4. Collect all **additional Python** and **non-Python** dependencies while doing this. 
  Consider fixing the version of each dependency to maintain reproducibility.
  5. Add dependencies to PyPI in a new file to `/extra_requirements`. The name of the file is secondary. However, add the dependencies as a list under the key `<new_benchmark>`.
  6. Add the remaining dependencies or steps necessary to run your benchmark in the docstring of your benchmark class
    (see, e.g. `hpobench/benchmarks/nas/nasbench_101.py`).
  7. Verify that everything works with, e.g.

```python
from hpobench.benchmarks.<type>.<new_benchmark> import <NewBenchmark>
b = <NewBenchmark>(<some_args>, rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, rng=1)
print(result_dict)
```

**Note:** Ideally, your benchmark behaves deterministic given a seed!

Now, you can create a PR marked as [WIP] and proceed with building a containerized version. 


## Create a containerized benchmark

  1. Create a container benchmark class `<NewBenchmark>` in `hpobench/container/benchmarks/<type>/<new_benchmark>.py` inheriting from the base class `AbstractBenchmarkClient` in `hpobench.container.client_abstract_benchmark`. The arguments `benchmark_name` and `container_name` should be assigned to `<NewBenchmark>`.
  <span style="color:red">Todo: What are benchmark_name + container_name for?</span>
  *Note: this are just a few lines of code, see, e.g. `hpobench/container/benchmarks/ml/xgboost_benchmark.py`).*.
  2. Copy `hpobench/container/recipes/Singularity.template` to  `hpobench/container/recipes/<type>/Singularity.<NewBenchmark>`.
  3. Modify the recipe and add your **additional Python** and **non-Python** dependencies collected above. Make sure you install the right dependencies with ```pip install .[<new_benchmark>]```.
  3. Test your container locally (see below).

Now, you can update your PR and let us know, so we can upload the container. Thanks.
  
## How to test your container locally

  1. `cd hpobench/container/benchmarks/recipes/<type>/Singularity.<NewBenchmark>` and change to following lines in the recipe:
  ```bash
    && git clone https://github.com/automl/HPOBench.git \
    && cd HPOBench \
    && git checkout development \
  ```
  to point to the branch/repo where your fork is on, e.g. `<branch_name>`.
  <span style="color:red">Todo: In Singularity.template is mentioned that you should not point to your branch.</span>
   
  2. Run `sudo singularity build <NewBenchmark> Singularity.<NewBenchmark>`
  3. Verify that everything works with:

```python
from hpobench.container.benchmarks.<type>.<new_benchmark> import <NewBenchmark>
b = <NewBenchmark>(container_source="./", container_name="<NewBenchmark>")
res = b.objective_function(configuration=b.get_configuration_space(seed=1).sample_configuration())
```

