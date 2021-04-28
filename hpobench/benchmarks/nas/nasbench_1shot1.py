"""
Interface to Benchmarks of Nasbench 1shot 1
https://github.com/automl/nasbench-1shot1/tree/master/nasbench_analysis/


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


Recommend: ``Python >= 3.6.0``

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .[nasbench_1shot1]

pip install git+https://github.com/google-research/nasbench.git@master
git clone https://github.com/automl/nasbench-1shot1/tree/master/nasbench_analysis/

3. Environment setup
====================

To use the nasbench_analysis package, add the path to this folder to your PATH variable.
```
export PATH=/Path/to/nasbench-1shot1:$PATH
```

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

0.0.1:
* First implementation

"""
import logging

from pathlib import Path
from typing import Union, Dict, Any, Tuple, List
from ast import literal_eval

import ConfigSpace as CS
import numpy as np
from nasbench import api
from nasbench.api import OutOfDomainError

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import NASBench_101DataManager
from hpobench.util import rng_helper

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1  # noqa
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2  # noqa
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3  # noqa
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3  # noqa

__version__ = '0.0.4'
logger = logging.getLogger('NasBench1shot1')


class NASBench1shot1BaseBenchmark(AbstractBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Baseclass for the nasbench 1shot1 benchmarks.
        Please install the benchmark first. Place the data under ``data_path``.

        Parameters
        ----------
        data_path : str, Path, None
            Path to the nasbench record. It is recommend to use the full record!
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """
        super(NASBench1shot1BaseBenchmark, self).__init__(rng=rng)
        data_manager = NASBench_101DataManager(data_path)
        self.api = data_manager.load()
        self.search_space = None

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           run_index: Union[int, Tuple, List, None] = (0, 1, 2),
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Query the NAS1shot1-benchmark using a given configuration and an epoch (=budget).
        Only data for the budgets 4, 12, 36, 108 are available.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        run_index : int, Tuple, None
            The nas benchmark has for each configuration-budget-pair results from 3 different runs.
            - If multiple `run_id`s are given as Tuple/List, the benchmark returns the mean over the given runs.
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
                train_accuracies
                test_accuracies
                valid_accuracies
                training_times
                fidelity : used fidelities in this evaluation
                data : additional data such as trainable parameters and used operations
        """
        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        run_index = self._check_run_index(run_index)
        configuration = self._parse_configuration(configuration)

        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        training_times = []
        additional = {}
        failure = False
        for run_id in run_index:
            data = self._query_benchmark(config=configuration, fidelity=fidelity, run_index=run_id)
            train_accuracies.append(data['train_accuracy'])
            valid_accuracies.append(data['validation_accuracy'])
            test_accuracies.append(data['test_accuracy'])
            training_times.append(data['training_time'])

            # Since those information are the same for all run ids, just store one of them.
            additional = {'trainable_parameters': data['trainable_parameters'],
                          'module_operations': data['module_operations']}
            failure = failure or ('info' in data and data['info'] == 'failure')

        return {'function_value': float(1 - np.mean(valid_accuracies)),
                'cost': float(np.sum(training_times)),
                'info': {'fidelity': fidelity,
                         'train_accuracies': train_accuracies,
                         'valid_accuracies': valid_accuracies,
                         'test_accuracies': test_accuracies,
                         'training_times': training_times,
                         'data': additional,
                         'failure': 'False' if not failure else 'True'
                         }
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the maximum available budget (108) and on all three seeds.

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
            function_value : test error on largest fidelity.
            cost : runtime
            info : Dict
                train_accuracies
                test_accuracies
                valid_accuracies
                training_times
                fidelity : used fidelities in this evaluation
                data : additional data such as trainable parameters and used operations
        """
        assert fidelity['budget'] == 108, 'Only test data for the 108th epoch is available.'
        result = self.objective_function(configuration=configuration, fidelity=fidelity, run_index=(0, 1, 2), rng=rng)
        result['function_value'] = float(1 - np.mean(result['info']['test_accuracies']))
        return result

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the NASBench1shot1.

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
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search',
                'references': ['@inproceedings{Zela2020NAS-Bench-1Shot1:, '
                               'title     = {NAS-Bench-1Shot1: '
                               '             Benchmarking and Dissecting One-shot Neural Architecture Search},'
                               'author    = {Arber Zela and Julien Siems and Frank Hutter},'
                               'booktitle = {International Conference on Learning Representations},'
                               'year      = {2020},'
                               'url       = {https://openreview.net/forum?id=SJx9ngStPH}}',
                               ],
                'code': 'https://github.com/automl/nasbench-1shot1',
                }

    def _check_run_index(self, run_index):

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

        return run_index

    def _query_benchmark(self, config: Dict, fidelity: Dict, run_index: int) -> Dict:

        adjacency_matrix, node_list = self.search_space.convert_config_to_nasbench_format(config)

        if isinstance(self, NASBench1shot1SearchSpace3Benchmark):
            node_list = [INPUT, *node_list, OUTPUT]
        else:
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)

        try:
            nasbench_data = self._query_api(model_spec=model_spec, run_index=run_index, epochs=int(fidelity['budget']))
        except api.OutOfDomainError:
            return {"trainable_parameters": 0,
                    "training_time": 0,
                    "train_accuracy": 0,
                    "validation_accuracy": 0,
                    "test_accuracy": 0,
                    "module_operations": 0,
                    "info": "failure"}

        nasbench_data.pop('module_adjacency')
        return nasbench_data

    def _query_api(self, model_spec, run_index: int, epochs=108, stop_halfway=False):
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
        if epochs not in self.api.valid_epochs:
            raise OutOfDomainError('invalid number of epochs, must be one of %s'
                                   % self.api.valid_epochs)

        fixed_stat, computed_stat = self.api.get_metrics_from_spec(model_spec)

        # MODIFICATION: Use the run index instead of the sampled one.
        # ORIGINAL CODE:
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

        self.api.training_time_spent += data['training_time']
        if stop_halfway:
            self.api.total_epochs_spent += epochs // 2
        else:
            self.api.total_epochs_spent += epochs

        return data

    def _parse_configuration(self, configuration: Dict):
        """
        Since the categorical hyperparameters are stored as strings (otherwise they would not be json serializable),
        we need to cast them back to type tuple.

        In the original configuration space all hyperparameters are either of type string or tuple.
        In the modified configuration space, the tuple hp are also strings.

        Parameters
        ----------
        configuration : Dict.

        Returns
        -------
        Dict - configuration with the correct types
        """
        # make sure that it is a dictionary and not a CS.Configuration.
        if isinstance(configuration, CS.Configuration):
            configuration = configuration.get_dictionary()

        return {k: literal_eval(v) if isinstance(v, str) and v[0] == '(' else v
                for k, v in configuration.items()}

    @staticmethod
    def _get_configuration_space(search_space: Any, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Helper function to pass a seed to the configuration space """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        original_cs = search_space.get_configuration_space()

        # The categorical hyperparameter of this benchmark consist of some tuple(tuple(int, int)). This is not
        # json serializable with the configspace serializer. Therefore, we cast it to a string.
        hps = []
        for hp in original_cs.get_hyperparameters():
            # the configspaces of this benchmark have only categorical hp
            # --> so they will all have the attribute 'default value'
            if isinstance(hp.default_value, tuple):
                hp = CS.CategoricalHyperparameter(hp.name,
                                                  choices=[str(choice) for choice in hp.choices],
                                                  default_value=str(hp.default_value))
            hps.append(hp)
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        cs.seed(seed)
        return cs


class NASBench1shot1SearchSpace1Benchmark(NASBench1shot1BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        super(NASBench1shot1SearchSpace1Benchmark, self).__init__(data_path=data_path, rng=rng)
        self.search_space = SearchSpace1()

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return NASBench1shot1BaseBenchmark._get_configuration_space(SearchSpace1(), seed)


class NASBench1shot1SearchSpace2Benchmark(NASBench1shot1BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        super(NASBench1shot1SearchSpace2Benchmark, self).__init__(data_path=data_path, rng=rng)
        self.search_space = SearchSpace2()

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return NASBench1shot1BaseBenchmark._get_configuration_space(SearchSpace2(), seed)


class NASBench1shot1SearchSpace3Benchmark(NASBench1shot1BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        super(NASBench1shot1SearchSpace3Benchmark, self).__init__(data_path=data_path, rng=rng)
        self.search_space = SearchSpace3()

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return NASBench1shot1BaseBenchmark._get_configuration_space(SearchSpace3(), seed)
