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
