# How to add a new benchmark step-by-step

## Create a local benchmark

Clone Hpolib2, switch to the development branch and create your own branch, then install hpolib2. 
with `pip install .`
```bash
git clone https://github.com/automl/HPOlib2.git
cd HPOlib2
git checkout development
git branch newBenchmark
git checkout newBenchmark
pip install .
```

Then: 

  1. Implement your new benchmark `hpolib/benchmarks/<type>/<name>` inheriting from the base class 
  `AbstractBenchmark` in `hpolib.abstract_benchmark`. Your benchmark should implement `__init__()`, 
  `get_configuration_space()`, `get_fidelity_space()`, `objective_function()` and `objective_function_test()`.
    A good example for this can be found in `hpolib/benchmarks/ml/xgboost_benchmark.py`
  3. If your benchmarks needs a dataset (e.g. for training a ml model), please also implement a DataManager, see e.g.
   `hpolib/util/openml_data_manager.py` with a `load()` method that downloads data once and reuses it for further calls.
  4. Collect all **additional Python** and **non-Python** dependencies while doing this. 
  Consider fixing the version of each dependency to maintain reproducibility.
  5. Add dependencies to PIPy in a new file to `/extra_requirements`
  6. Add the remaining dependencies or steps necessary to run your benchmark in the docstring of your benchmark class
    (see, e.g. `hpolib/benchmarks/nas/nasbench_101.py`).
  7. Verify that everything works with, e.g.

```python
from hpolib.benchmarks.<type>.<newbenchmark> import <NewBenchmark>
b = <NewBenchmark>(<some_args>, rng=1)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config, rng=1)
print(result_dict)
```

**Note:** Ideally, your benchmark behaves deterministic given a seed!

Now, you can create a PR marked as [WIP] and proceed with building a containerized version. 

## Create a containerized benchmark

  1. Create a container benchmark class in `hpolib/container/benchmarks/<type>/<name>` inheriting from the 
  base class `AbstractBenchmarkClient` in `hpolib.container.client_abstract_benchmark`. 
  Note: this are just a few lines of code, see, e.g. `hpolib/container/benchmarks/ml/xgboost_benchmark.py`)
  2. Copy `hpolib/container/recipes/Singularity.template` to  `hpolib/container/recipes/<type>/name`
  3. Modify the recipe and add your **additional Python** and **non-Python** dependencies collected above. 
  3. Test your container locally (see below)

Now, you can update your PR and let us know, so we can upload the container to Sylabs. Thanks.
  
## How to test your container locally

  1. `cd hpolib/container/benchmarks/recipes/<type>` and change to following lines in the recipe:
  ```bash
    && git clone https://github.com/automl/HPOlib2.git \
    && cd HPOlib2 \
    && git checkout development \
```
   to point to the branch/repo where your fork is on, e.g. `newBenchmark`
  2. Run `sudo singularity build <newBenchmark> Singularity.<newBenchmark>`
  3. Verify that everything works with:

```python
from hpolib.container.benchmarks.<type>.<newbenchmark> import <NewBenchmark>
b = <NewBenchmark>(container_source="./", container_name="newBenchmark")
res = b.objective_function(configuration=b.get_configuration_space(seed=1).sample_configuration())
```
