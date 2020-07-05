"""
Interface to NasBench101 for Hyperparameter Optimization and Neural Architecture Search

https://github.com/automl/nas_benchmarks

How to install:
---------------

1) Download data
```
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
```
Remark: it is important to select the full tfrecord to perform multi-fidelity optimization.

2) Clone + Install
```
pip install git+https://github.com/google-research/nasbench.git@master
pip install git+https//github.com/automl/nas_benchmarks.git@master

pip install --upgrade "tensorflow>=1.12.1,<=1.15"

Notes:
------
THOSE BENCHMARKS CONTAIN ONLY VALUES FOR THE EPOCHS 4, 12, 36 and 108. FOR ALL OTHERS IT ALWAYS RETURN AN ACCURACY OF 0.
```
"""

from pathlib import Path
from typing import Union, Dict

import ConfigSpace
import numpy as np
from tabular_benchmarks.nas_cifar10 import NASCifar10

import hpolib.util.rng_helper as rng_helper
from hpolib.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'

MAX_EDGES = 9
VERTICES = 7


class NASCifar10BaseBenchmark(AbstractBenchmark):
    def __init__(self, benchmark: NASCifar10, data_path: Union[Path, str, None] = "./",
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Baseclass for the tabular benchmarks https://github.com/automl/nas_benchmarks/tree/master/tabular_benchmarks.
        Please install the benchmark first. Place the data under ``data_path``.

        Parameters
        ----------
        benchmark : NASCifar10
            Type of the benchmark to use. Don't call this class directly. Instantiate via subclasses (see below).
        data_path : str, Path, None
            Path to the folder, which contains the downloaded tabular benchmarks.
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """

        super(NASCifar10BaseBenchmark, self).__init__(rng=rng)

        self.benchmark = benchmark
        self.data_path = data_path

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           budget: Union[int, None] = 108,
                           reset: bool = True,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Query the NAS-benchmark using a given configuration and a epoch (=budget).

        Parameters
        ----------
        configuration : Dict
        budget : int, None
            Query the tabular benchmark at the `budget`-th epoch. Defaults to the maximum budget (108).
        reset : bool
            Reset the internal memory of the benchmark. Should not have an effect.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict
        """
        # TODO: Is a assertion too restrictive? If not valid, object_func already returns (1, 0).
        # assert budget in [4, 12, 36, 108], f'This benchmark supports only budgets [4, 12, 36, 108], but was {budget}'

        if reset:
            self.benchmark.reset_tracker()

        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        # Returns (valid_error_rate: 1, runtime: 0) if it is invalid. E.g. config not ok or budget not in 4 12 36 108
        valid_error_rate, runtime = self.benchmark.objective_function(config=configuration, budget=budget)
        return {'function_value': valid_error_rate,
                'cost': runtime,
                }

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration: Dict, rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the maximum available budget.

        Parameters
        ----------
        configuration : Dict
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict
        """
        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        test_error, runtime = self.benchmark.objective_function(config=configuration, budget=108)
        return {'function_value': test_error, 'cost': runtime}

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:

        raise NotImplementedError

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search',
                'references': ['Aaron Klein, Frank Hutter',
                               'Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization',
                               'https://arxiv.org/abs/1905.04970',
                               'https://github.com/automl/nas_benchmarks'],
                }


class NASCifar10ABenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks.nas_cifar10 import NASCifar10A
        benchmark = NASCifar10A(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10ABenchmark, self).__init__(benchmark=benchmark, data_path=data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        """
        get_configuration_space method from:
        https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
        (Aaron Klein).
        Copied to pass a seed to the ConfigurationSpace object.

        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            ConfigSpace.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        # OLD
        # from tabular_benchmarks.nas_cifar10 import NASCifar10A
        # return NASCifar10A.get_configuration_space()

        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = ConfigSpace.ConfigurationSpace(seed=seed)

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs


class NASCifar10BBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks.nas_cifar10 import NASCifar10B
        benchmark = NASCifar10B(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10BBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        """
        get_configuration_space method from:
        https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
        (Aaron Klein).
        Copied to pass a seed to the ConfigurationSpace object.

        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            ConfigSpace.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        # OLD
        # from tabular_benchmarks.nas_cifar10 import NASCifar10B
        # return NASCifar10B.get_configuration_space()
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = ConfigSpace.ConfigurationSpace(seed=seed)

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        cat = [i for i in range((VERTICES * (VERTICES - 1)) // 2)]
        for i in range(MAX_EDGES):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, cat))
        return cs


class NASCifar10CBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks.nas_cifar10 import NASCifar10C
        benchmark = NASCifar10C(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10CBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        """
        get_configuration_space method from:
        https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
        (Aaron Klein).
        Copied to pass a seed to the ConfigurationSpace object.

        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            ConfigSpace.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        # OLD:
        # from tabular_benchmarks.nas_cifar10 import NASCifar10C
        # return NASCifar10C.get_configuration_space()
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = ConfigSpace.ConfigurationSpace(seed=seed)

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("num_edges", 0, MAX_EDGES))

        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("edge_%d" % i, 0, 1))
        return cs
