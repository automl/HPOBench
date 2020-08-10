# How to add a new benchmark step-by-step

## Create a local benchmark

  1. Clone Hpolib2, switch to the development branch and create your own branch, then install hpolib2. 
Note that with `pip install .`

```bash
git clone https://github.com/automl/HPOlib2.git
cd HPOlib2
git checkout development
git branch newBenchmark
git checkout newBenchmark
pip install .
```

  2. Implement your new benchmark `hpolib/benchmarks/<type>/<name>` inheriting from the base class 
  `AbstractBenchmark` in `hpolib.abstract_benchmark`
 
  3. Collect **all additional Python** and **non-Python** dependencies while doing this. 
  Consider fixing the version of each dependency to maintain reproducibility.
  4. Add dependencies to PiPy in a new file to `/extra_requirements`
  5. Add the remaining dependencies or steps necessary to run your benchmark in the docstring of your benchmark class.

## Create a containerized benchmark

  1. Create a container benchmark class in `hpolib/container/benchmarks/<type>/<name>` inheriting from the 
  base class `AbstractBenchmarkClient` in `hpolib.container.client_abstract_benchmark` (note: this is just copy/paste from existing classes)
  2. Copy `hpolib/container/recipes/Singularity.template` to  `hpolib/container/recipes/<type>/name`
  3. Create a pull request marked as [WIP]
  
## How to test your container locally

  1. `cd hpolib/container/benchmarks/recipes/<type>` and change to following line in the recipe:
  ```bash
    && git checkout development \
```
   to point to the branch where your pull request is, e.g. `newBenchmark`
  2. Run `sudo singularity build <newBenchmark> Singularity.<newBenchmark>`
  3. Verify that everything works with
  ```python
from hpolib.container.benchmarks.<type>.new_benchmark import newBenchmark
b = newBenchmark(container_source="./", container_name="newBenchmark")
res = b.objective_function(configuration=b.get_configuration_space(seed=1).sample_configuration())
```
  4. Finalize your pull request and let us know, so we can upload the container to Sylabs. Thanks.