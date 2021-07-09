# How to add a new benchmark step-by-step

Please read and understand the [README.md](https://github.com/automl/HPOBench/blob/master/README.md) first before adding a new benchmark.

## Placeholders for your benchmark

- `<type>`: Category of the benchmark, e.g. ml (machine learning)
- `<new_benchmark>`: Filename of the benchmark*, e.g. svm
- `<NewBenchmark>`: Classname of the benchmark*, e.g. SupportVectorMachine

`*`: has to be unique across all available benchmarks.


## Create a local benchmark

[Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) HPOBench and clone it to your machine. Switch to the development branch and create your own branch. Then install HPOBench with `pip install .`
```bash
git clone https://github.com/<your_github_name>/HPOBench.git
cd HPOBench
git checkout development
git branch <new_benchmark>
git checkout <new_benchmark>
pip install .
```

Then: 
  1. Implement your new benchmark class `<NewBenchmark>` in `hpobench/benchmarks/<type>/<new_benchmark>.py` inheriting from the base class 
  `AbstractBenchmark` in `hpobench.abstract_benchmark`. Your benchmark should implement `__init__()`, 
  `get_configuration_space()`, `get_fidelity_space()`, `objective_function()` and `objective_function_test()`.
  3. If your benchmarks needs a dataset (e.g. for training a ml model), please also implement a DataManager, see e.g.
   `hpobench/util/openml_data_manager.py` with a `load()` method that downloads data once and reuses it for further calls.
  4. Collect all **additional Python** and **non-Python** dependencies while doing this. Make sure to also note the exact version of each dependency.
  5. Add dependencies installable via `pip` in a new file to `/extra_requirements`. While the name of the file is secondary, please choose a descriptive name. Add the dependencies as a list under the key `<new_benchmark>`.
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

  1. Create a container benchmark class `<NewBenchmark>` in `hpobench/container/benchmarks/<type>/<new_benchmark>.py` inheriting from the base class `AbstractBenchmarkClient` in `hpobench.container.client_abstract_benchmark`. The arguments `benchmark_name` and `container_name` should be assigned to `<NewBenchmark>` and `<new_benchmark>`, respectively.
  2. Copy `hpobench/container/recipes/Singularity.template` to `hpobench/container/recipes/<type>/Singularity.<NewBenchmark>`.
  3. Make sure you install the right dependencies within the file ```pip install .[<new_benchmark>]```.
  3. Test your container locally (see below).

Now, you can update your PR and let us know s.t. we can upload the container. Thanks.
  
## How to test your container locally

  1. Switch into the folder `hpobench/container/recipes/<type>`, open the file `Singularity.<NewBenchmark>` and change the following lines in the recipe
  ```bash
    && git clone https://github.com/automl/HPOBench.git \
    && cd HPOBench \
    && git checkout development \
  ```
  to point to the repo and branch where your fork is on:
  ```bash
    && git clone https://github.com/<your_github_name>/HPOBench.git \
    && cd HPOBench \
    && git checkout <new_benchmark> \
  ```

  2. Run `sudo singularity build <new_benchmark> Singularity.<NewBenchmark>`.
  3. Verify everything with:

```python
from hpobench.container.benchmarks.<type>.<new_benchmark> import <NewBenchmark>
b = <NewBenchmark>(container_source="./", container_name="<new_benchmark>")
res = b.objective_function(configuration=b.get_configuration_space(seed=1).sample_configuration())
```


