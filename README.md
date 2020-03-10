# HPOlib3

## Installation
Installation via \
``` pip install <path_to_hpolib3_repository>[extra requirements] ```\
Extra requirements could be 'xgboost', 'singularity'.

#### XGBoost Example
Then, install special dependencies for this benchmark with \
``` pip install "<path_to_hpolib3_repository>[xgboost]"  ```

To run the XGBoost Example: \
```python3 ./HPOlib3/examples/XGBoost_local.py```

#### Example with container
``` pip install "<path_to_hpolib3_repository>[xgboost,singularity]"  ```\

To run the Example: \
```python3 ./HPOlib3/examples/XGBoost_with_container.py```

#### Use singularity on Cluster:
This works currently only on 'kisbat3'. To use the the singularity version 3.5, first you have 
to set the following path:\
```export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH```

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
  This happens, if you haven't exported the path to singularity3.5 (see above).

