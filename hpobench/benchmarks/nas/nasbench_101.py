"""
Interface to NasBench101 for Hyperparameter Optimization and Neural Architecture Search

https://github.com/automl/nas_benchmarks

How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

1. Download data
================

The data will be downloaded automatically.
Note: However, if you use the benchmark locally, you can specify also the data directory (path to the folder, where the
nasbench_full.tfrecord is) by hand.

In this case you can download the data with the following command.

```
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
```
Remark: it is important to select the full tf record and not the 'only_108' record to perform multi-fidelity
optimization.

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .[nasbench_101]

pip install git+https://github.com/google-research/nasbench.git@master
pip install git+https://github.com/automl/nas_benchmarks.git@master
```

Notes:
------
Benchmarks in NASBench101 only contain epochs 4, 12, 36 and 108.
Querying another epoch, e.g. 5, raises an assertion.

Changelog:
==========
0.0.4
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.3:
* Standardize the structure of the meta information

0.0.2:
* The objective function takes as input now the parameter run_index. Allowed values are Tuple(0-2), 0, 1, 2, None.
  This value specifies which seeds are used. The user can specify a single index or a tuple with indices.
  If the user wants to use a randomly drawn run_index, they can simply set the value explicitly to None.
* Fix a bug in NASCifar10CBenchmark

0.0.1:
* First implementation


"""
import logging

from pathlib import Path
from typing import Union, Dict, Any, Tuple, List

import ConfigSpace as CS
import numpy as np
from tabular_benchmarks.nas_cifar10 import NASCifar10
from nasbench import api
from nasbench.api import OutOfDomainError
from nasbench.lib import graph_util

from hpobench import config_file
import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import NASBench_101DataManager

__version__ = '0.0.4'
logger = logging.getLogger('NasBench101')

MAX_EDGES = 9
VERTICES = 7
DEFAULT_API_FILE = config_file.data_dir / "nasbench_101"


class NASCifar10BaseBenchmark(AbstractBenchmark):
    def __init__(self, benchmark: NASCifar10,
                 data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """
        Baseclass for the tabular benchmarks https://github.com/automl/nas_benchmarks/tree/master/tabular_benchmarks.
        Please install the benchmark first. Place the data under ``data_path``.

        Parameters
        ----------
        benchmark : NASCifar10
            Type of the benchmark to use. Don't call this class directly. Instantiate via subclasses (see below).
        data_path : str, Path, None
            Path to the folder, which contains the downloaded file nasbench_full.tfrecord.
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """

        super(NASCifar10BaseBenchmark, self).__init__(rng=rng)

        self.benchmark = benchmark
        self.data_path = data_path

    def _query_benchmark(self, config: Dict, run_index: int, budget: int = 108) -> Dict:
        raise NotImplementedError

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           run_index: Union[int, Tuple, None] = (0, 1, 2),
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Query the NAS-benchmark using a given configuration and a epoch (=budget).

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        run_index : int, Tuple, None
            The nas benchmark has for each configuration-budget-pair results from 3 different runs.
            - If multiple `run_id`s are given as Tuple, the benchmark returns the mean over the given runs.
            - By default (no parameter is specified) all runs are used. A specific run can be chosen by setting the
              `run_id` to a value from [0, 3]. While the performance is averaged across the `run_index`, the costs are
              the sum of the runtime per `run_index`.
            - When this value is explicitly set to `None`, the function will use a random seed.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation error
            cost : runtime
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        if isinstance(run_index, int):
            assert 0 <= run_index <= 2, f'run_index must be in [0, 2], not {run_index}'
            run_index = (run_index, )
        elif isinstance(run_index, (Tuple, List)):
            assert 0 < len(run_index) <= 3, 'run_index must not be empty'
            assert min(run_index) >= 0 and max(run_index) <= 2, \
                f'all run_index values must be in [0, 2], but were {run_index}'
            if len(set(run_index)) != len(run_index):
                logger.debug('There are some values more than once in the run_index. We remove the redundant entries.')
                run_index = tuple(set(run_index))
        elif run_index is None:
            logger.debug('The run index is explicitly set to None! A random seed will be selected.')
            run_index = tuple(self.rng.choice((0, 1, 2), size=1))
        else:
            raise ValueError(f'run index must be one of Tuple or Int, but was {type(run_index)}')

        self.benchmark.reset_tracker()

        # Returns (valid_accuracy: 0, runtime: 0) if it is invalid, e.g. config not valid or
        # budget not in 4 12 36 108
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        training_times = []
        additional = {}

        for run_id in run_index:
            data = self._query_benchmark(config=configuration, budget=fidelity['budget'], run_index=run_id)

            train_accuracies.append(data['train_accuracy'])
            valid_accuracies.append(data['validation_accuracy'])
            test_accuracies.append(data['test_accuracy'])
            training_times.append(data['training_time'])

            # Since those information are the same for all run ids, just store one of them.
            additional = {'trainable_parameters': data['trainable_parameters'],
                          'module_operations': data['module_operations']}

        return {'function_value': float(1 - np.mean(valid_accuracies)),
                'cost': float(np.sum(training_times)),
                'info': {'fidelity': fidelity,
                         'train_accuracies': train_accuracies,
                         'valid_accuracies': valid_accuracies,
                         'test_accuracies': test_accuracies,
                         'training_times': training_times,
                         'data': additional
                         }
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the maximum available budget.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is
            possible to pass a parameter ``rng`` as 'int' or 'np.random.RandomState' to this
            function. If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : test error
            cost : runtime
            info : Dict
                fidelity : used fidelities in this evaluation
        """

        result = self.objective_function(configuration=configuration, fidelity=fidelity, run_index=(0, 1, 2), rng=rng)
        result['function_value'] = float(1 - np.mean(result['info']['test_accuracies']))

        return result

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search',
                'references': ['@article{klein2019tabular,'
                               'title   = {Tabular benchmarks for joint architecture and hyperparameter optimization},'
                               'author  = {Klein, Aaron and Hutter, Frank},'
                               'journal = {arXiv preprint arXiv:1905.04970},'
                               'year    = {2019}}',
                               'https://arxiv.org/abs/1905.04970',
                               ],
                'code': 'https://github.com/automl/nas_benchmarks',
                }

    @staticmethod
    def _get_configuration_space(benchmark: Any, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Helper function to pass a seed to the configuration space """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = benchmark.get_configuration_space()
        cs.seed(seed)
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the NAS Benchmark 101.

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.OrdinalHyperparameter('budget', sequence=[4, 12, 36, 108], default_value=108)
        ])

        return fidel_space

    @staticmethod
    def _try_download_api_file(save_to: Union[Path, str, None]):
        data_manager = NASBench_101DataManager(save_to)
        data_manager.download()
        return data_manager.save_dir


class NASCifar10ABenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        data_path = self._try_download_api_file(data_path)

        from tabular_benchmarks.nas_cifar10 import NASCifar10A
        benchmark = NASCifar10A(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10ABenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng, **kwargs)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Return the configuration space for the NASCifar10A benchmark.
        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            CS.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        from tabular_benchmarks.nas_cifar10 import NASCifar10A
        return NASCifar10BBenchmark._get_configuration_space(NASCifar10A, seed)

    def _query_benchmark(self, config: Dict, run_index: int, budget: int = 108) -> Dict:
        """
        Copied from the 'objective_function' from nas_cifar10.py
        We adapted the file in such a way, that the complete result is returned. The original implementation returns
        only the validation error. Now, it can also return the test loss for a given configuration.

        Parameters
        ----------
        config : Dict
        run_index : int
            Specifies the seed to use. Can be one of 0, 1, 2.
        budget : int
            The number of epochs. Must be one of: 4 12 36 108. Otherwise a accuracy of 0 is returned.

        Returns
        -------
        Dict
        """

        failure = {"test_accuracy": 0, "train_accuracy": 0, "validation_accuracy": 0, "training_time": 0,
                   "info": "failure", "trainable_parameters": 0, "module_operations": 0}

        if self.benchmark.multi_fidelity is False:
            assert budget == 108

        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = config["edge_%d" % i]

        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.benchmark.record_invalid(config, 1, 1, 0)
            return failure

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)

        try:
            data = modified_query(self.benchmark, run_index=run_index, model_spec=model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.benchmark.record_invalid(config, 1, 1, 0)
            return failure

        self.benchmark.record_valid(config, data, model_spec)

        # We dont need this field.
        data.pop('module_adjacency')

        return data


class NASCifar10BBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        data_path = self._try_download_api_file(data_path)

        from tabular_benchmarks.nas_cifar10 import NASCifar10B
        benchmark = NASCifar10B(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10BBenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng, **kwargs)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Return the configuration space for the NASCifar10B benchmark.
        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            CS.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        from tabular_benchmarks.nas_cifar10 import NASCifar10B
        return NASCifar10BBenchmark._get_configuration_space(NASCifar10B, seed)

    def _query_benchmark(self, config: Dict, run_index: int, budget: int = 108) -> Dict:
        """
        Copied from the 'objective_function' from nas_cifar10.py
        We adapted the file in such a way, that the complete result is returned. The original implementation returns
        only the validation error. Now, it can also return the test loss for a given configuration.

        Parameters
        ----------
        config : Dict
        budget : int
            The number of epochs. Must be one of: 4 12 36 108. Otherwise a accuracy of 0 is returned.

        Returns
        -------
        Dict
        """
        failure = {"test_accuracy": 0, "train_accuracy": 0, "validation_accuracy": 0, "training_time": 0,
                   "info": "failure", "trainable_parameters": 0, "module_operations": 0}

        if self.benchmark.multi_fidelity is False:
            assert budget == 108

        bitlist = [0] * (VERTICES * (VERTICES - 1) // 2)
        for i in range(MAX_EDGES):
            bitlist[config["edge_%d" % i]] = 1
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit

        matrix = np.fromfunction(graph_util.gen_is_edge_fn(out),
                                 (VERTICES, VERTICES),
                                 dtype=np.int8)
        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.benchmark.record_invalid(config, 1, 1, 0)
            return failure

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = modified_query(self.benchmark, run_index=run_index, model_spec=model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.benchmark.record_invalid(config, 1, 1, 0)
            return failure

        self.benchmark.record_valid(config, data, model_spec)

        # We dont need this field.
        data.pop('module_adjacency')

        return data


class NASCifar10CBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        data_path = self._try_download_api_file(data_path)

        from tabular_benchmarks.nas_cifar10 import NASCifar10C
        benchmark = NASCifar10C(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10CBenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng, **kwargs)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Return the configuration space for the NASCifar10C benchmark.
        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            CS.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        from tabular_benchmarks.nas_cifar10 import NASCifar10C
        return NASCifar10BBenchmark._get_configuration_space(NASCifar10C, seed)

    def _query_benchmark(self, config: Dict, run_index: int, budget: int = 108) -> Dict:
        """
        Copied from the 'objective_function' from nas_cifar10.py
        We adapted the file in such a way, that the complete result is returned. The original implementation returns
        only the validation error. Now, it can also return the test loss for a given configuration.

        Parameters
        ----------
        config : Dict
        budget : int
            The number of epochs. Must be one of: 4 12 36 108. Otherwise a accuracy of 0 is returned.

        Returns
        -------
        Dict
        """
        # Unify the return value to a dictionary.
        failure = {"test_accuracy": 0, "train_accuracy": 0, "validation_accuracy": 0, "training_time": 0,
                   "info": "failure", "trainable_parameters": 0, "module_operations": 0}

        if self.benchmark.multi_fidelity is False:
            assert budget == 108

        edge_prob = []
        for i in range(VERTICES * (VERTICES - 1) // 2):
            edge_prob.append(config["edge_%d" % i])

        idx = np.argsort(edge_prob)[::-1][:config["num_edges"]]
        binay_encoding = np.zeros(len(edge_prob))
        binay_encoding[idx] = 1
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = binay_encoding[i]

        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.benchmark.record_invalid(config, 1, 1, 0)
            return failure

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = modified_query(self.benchmark, run_index=run_index, model_spec=model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.benchmark.record_invalid(config, 1, 1, 0)
            return failure

        self.benchmark.record_valid(config, data, model_spec)

        # We dont need this field.
        data.pop('module_adjacency')

        return data


def modified_query(benchmark, model_spec, run_index: int, epochs=108, stop_halfway=False):
    """
    NOTE:
    Copied from https://github.com/google-research/nasbench/blob/b94247037ee470418a3e56dcb83814e9be83f3a8/nasbench/api.py#L204-L263  # noqa
    We changed the function in such a way that we now can specified the run index (index of the evaluation) which was
    in the original code sampled randomly.

    OLD DOCSTRING:
    Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.training_time_spent, and self.total_epochs_spent.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.

    Args:
      model_spec: ModelSpec object.
      epochs: number of epochs trained. Must be one of the evaluated number of
        epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    if epochs not in benchmark.dataset.valid_epochs:
        raise OutOfDomainError('invalid number of epochs, must be one of %s'
                               % benchmark.dataset.valid_epochs)

    fixed_stat, computed_stat = benchmark.dataset.get_metrics_from_spec(model_spec)

    # MODIFICATION: Use the run index instead of the sampled one.
    # sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    computed_stat = computed_stat[epochs][run_index]

    data = {}
    data['module_adjacency'] = fixed_stat['module_adjacency']
    data['module_operations'] = fixed_stat['module_operations']
    data['trainable_parameters'] = fixed_stat['trainable_parameters']

    if stop_halfway:
        data['training_time'] = computed_stat['halfway_training_time']
        data['train_accuracy'] = computed_stat['halfway_train_accuracy']
        data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
        data['test_accuracy'] = computed_stat['halfway_test_accuracy']
    else:
        data['training_time'] = computed_stat['final_training_time']
        data['train_accuracy'] = computed_stat['final_train_accuracy']
        data['validation_accuracy'] = computed_stat['final_validation_accuracy']
        data['test_accuracy'] = computed_stat['final_test_accuracy']

    benchmark.dataset.training_time_spent += data['training_time']
    if stop_halfway:
        benchmark.dataset.total_epochs_spent += epochs // 2
    else:
        benchmark.dataset.total_epochs_spent += epochs

    return data
