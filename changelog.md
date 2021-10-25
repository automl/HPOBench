# 0.0.11

# 0.0.10
  * Cartpole Benchmark Version 0.0.4:
    Fix: Pass the hp `entropy_regularization` to the PPO Agent. 
    Increase the lower limit of `likelihood_ratio_clipping` from 0 to 10e-7 (0 is invalid.)
  * Tabular Benchmark (NAS) Version 0.0.5 + NAS201 Version 0.0.5:
    We add for each benchmark in nas/tabular_benchmarks and nas/nasbench_201 a new benchmark class with a modified fidelity space. The new benchmark are called _Original_, e.g. _SliceLocalizationBenchmarkOriginal_ compared to _SliceLocalizationBenchmark_
    These new benchmarks have the same fidelity space as used in previous experiments by [DEHB](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/21-IJCAI-DEHB.pdf) and [BOHB](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf).
    Specifically, we increase the lowest fidelity from 1 to 3 for nas/tabular_benchmarks and from 1 to 12 for nas/nasbench_201. The upper fidelity and the old benchmarks remain unchanged.
  
# 0.0.9
  * Add new Benchmarks: Tabular Benchmarks.
    Provided by @Neeratyoy. 
  * New Benchmark: ML Benchmark Class
    This new benchmark class offers a unified interface for XGB, SVM, MLP, HISTGB, RF, LR benchmarks operating on OpenML 
    tasks.
    Provided by @Neeratyoy.  
  * This version is the used for the paper: 
    "HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO" (Eggensperger et al.)    
    https://openreview.net/forum?id=1k4rJYEwda-
    
# 0.0.8
  * Improve container integration
    The containers had some problems when the file system was read-only. In this case, the home directory, which contains the 
    hpobenchrc file, was not mounted, so the init of the containers failed. To circumvent this problem, we make multiple
    changes: 
    * We introduce an option to mount additional folder. This might be helpful when working on a cluster where the home 
      directory is not available in the computation process.
    * We also change the configuration file. The container does not read the yaml file anymore. Instead we bind the 
      cache dir, data dir and socket dir into the container and let the container use them directly. We also remove the 
      global data directory and use only the data dir from now onwards.
  * Add the surrogate SVM on MNIST benchmark from the BOHB paper.
  * ParamNetBenchmark: 
    Suppress some unuseful warnings and introduce a new param net benchmark that has a reduced search space.
  * Add Clean Up script: 
    See the documentation for more information. You can clean all caches and container files by calling:
    `python util/clean_up_scripts.py --clear_all`
  * Improve Information on how to add a new benchmark
  * Fix an error in PyBnn Benchmark:
    We introduce the benchmark version 0.0.4.  
    In this new version, we prevent infinity values in the nll loss when the predicted variance
    is 0. We set the predicted variance before computing the log to max(var, 1e-10)         
# 0.0.7
  * Fix an error in the NASBench1shot1 Benchmark (SearchSpace3).
  * Improve the behavior when a benchmark container is shut down.
  * Fix an error in PyBnn Benchmark:
    Set the minimum number of steps to burn to 1. 
  * Fix an error in ParamNetOnTime Benchmarks:
    The maximum budget was not properly determined.
  * Move Container to Gitlab:
    Add support for communicating with the gitlab registry. We host the container now on \
    https://gitlab.tf.uni-freiburg.de/muelleph/hpobench-registry/container_registry
  * Update the version check:  Instead of requesting the same version, we check if the configuration file version and the
    hpobench version are in the same partition. Each hpobench version that has a compatible configuration file definition
    is in the same distribution.
  * Introduce a new version of the XGBoostBenchmark: A benchmark with an additional parameter, `booster`.
  * New Parameter Container Tag:
    The container-benchmark interface takes as input an optional container tag. by specifying this parameter, 
    different container are downloaded from the registry.
  * Improve the procedure to find a container on the local filesystem:
    If a container_source is given, we check if it is either the direct address of a container on the filesystem 
    or a directory. In case, it is a directory, we try to find the correct container in this directory by appending the 
    container_tag to the container_name. (<container_source>/<container_name_<container_tag>)
    
# 0.0.6
  * Add NasBench1shot1 Benchmark
  * Add info about incumbents for nasbench201 to its docstrings. 
  * XGB and SVM's `get_meta_information()`-function returns now information about the used data set split.
  * Simplify the wrapper. Remove the support for configurations as lists and arrays. 
  * Enforce the correct class-interfaces with the pylint package. To deviate from the standard interface,
    you have to explicitly deactivate the pylint error. 
  * Nas1shot1 and Nas101 take as as input parameter now a seed.
  * The config file is now based on yaml. Also, it automatically raises a warning if the configuration file-version 
    does not match the HPOBench-version.
    
# 0.0.5
  * Rename package to HPOBench
  * Add BNN (pybnn) benchmark
  * Update returned loss values for nasbench101 and tabular benchmarks
  * Updat returned loss values for nasbench201 as well its data
  * Nasbench201 is now 1 indexed instead of 0.
  * Add MinMaxScaler to SVM Benchmark's preprocessing
  * Add further tests
  
# 0.0.4
  * improve test coverage
  * update HowToAddANewBenchmark.md
  * Add SVM benchmark working on OpenML data
  * Add nasbench201 benchmark
  
# 0.0.3
  * improve forwarding exceptions in containerized benchmarks
  * allow to set debug level with env variable
  * rename everything to HPOlib2 

# 0.0.2
  * add first set of benchmarks 

# 0.0.1
 * initial release
