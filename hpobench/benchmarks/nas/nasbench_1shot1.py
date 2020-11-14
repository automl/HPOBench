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
export PATH=/Path/to/nasbench-1shot1-directory:$PATH
```
"""

from pathlib import Path
from typing import Union, Dict, Any
from ast import literal_eval

import ConfigSpace as CS
import numpy as np
from nasbench import api
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import NASBench_101DataManager

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1  # noqa
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2  # noqa
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3  # noqa
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3  # noqa

__version__ = '0.0.1'


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

    def _query_benchmark(self, config: Dict, fidelity: Dict) -> Dict:
        adjacency_matrix, node_list = self.search_space.convert_config_to_nasbench_format(config)
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = self.api.query(model_spec, epochs=int(fidelity['budget']))

        info = {'trainable_parameters': nasbench_data['trainable_parameters'],
                'training_time': nasbench_data['training_time'],
                'train_accuracy': nasbench_data['train_accuracy'],
                'validation_accuracy': nasbench_data['validation_accuracy'],
                'test_accuracy': nasbench_data['test_accuracy'],
                'fidelity': fidelity}

        return info

    def _parse_configuration(self, configuration: Dict):
        """
        Since the categorical hyperparameters are stored as strings (otherwise they are not json serializable),
        we need to cast them back to tuple.

        In the original configuration space all hyperparameters are of either of type string or tuple.
        In the modified, also the tuple hp are strings. A tuple hyperparameter is indicated here by a opening bracket.

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

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Query the NAS-benchmark using a given configuration and a epoch (=budget).
        Only data for the budgets 4, 12, 36, 108 are available.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
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
                trainable_parameters:
                training_time
                train_accuracy
                validation_accuracy
                test_accuracy
                fidelity : used fidelities in this evaluation
        """
        configuration = self._parse_configuration(configuration)

        info = self._query_benchmark(configuration, fidelity)

        return {'function_value': 1 - info['validation_accuracy'],
                'cost': info['training_time'],
                'info': info}

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the maximum available budget (108).

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
                trainable_parameters:
                training_time
                train_accuracy
                validation_accuracy
                test_accuracy
                fidelity : used fidelities in this evaluation
        """

        assert fidelity['budget'] == 108, 'Only test data for the 108th epoch is available. '

        configuration = self._parse_configuration(configuration)
        info = self._query_benchmark(configuration, fidelity)

        return {'function_value': 1 - info['test_accuracy'],
                'cost': info['training_time'],
                'info': info}

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': '',
                'references': ['Arber Zela and Julien Siems and Frank Hutter',
                               'NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search',
                               '@inproceedings{Zela2020NAS-Bench-1Shot1:, '
                               'title={NAS-Bench-1Shot1: '
                               '       Benchmarking and Dissecting One-shot Neural Architecture Search},'
                               'author={Arber Zela and Julien Siems and Frank Hutter},'
                               'booktitle={International Conference on Learning Representations},'
                               'year={2020},'
                               'url={https://openreview.net/forum?id=SJx9ngStPH}}'
                               'https://github.com/automl/nasbench-1shot1'],
                }

    @staticmethod
    def _get_configuration_space(search_space: Any, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Helper function to pass a seed to the configuration space """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        original_cs = search_space.get_configuration_space()

        # The categorical hyperparameter of this benchmark consist of some tuple(tuple(int, int)). This is not by
        # json serializable with the configspace. Therefore, we cast it to a string.
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
