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
import ConfigSpace as CS
import numpy as np
from tabular_benchmarks.nas_cifar10 import NASCifar10

import hpolib.util.rng_helper as rng_helper
from hpolib.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'


class NASCifar10BaseBenchmark(AbstractBenchmark):
    def __init__(self, benchmark: NASCifar10, data_path: Union[Path, str, None] = "./",
                 rng: Union[np.random.RandomState, int, None] = None):

        super(NASCifar10BaseBenchmark, self).__init__(rng=rng)

        self.benchmark = benchmark
        self.data_path = data_path


    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           budget: Union[int, None] = 108,
                           reset: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        # TODO: Is a assertion too restrictive? If not valid, object_func already returns (1, 0).
        # assert budget in [4, 12, 36, 108], f'This benchmark supports only budgets [4, 12, 36, 108], but was {budget}'

        # TODO: do we want a new benchmark each time?
        if reset:
            self.benchmark.reset_tracker()

        self.rng = rng_helper.get_rng(rng)

        # Returns (valid_error_rate: 1, runtime: 0) if it is invalid. E.g. config not ok or budget not in 4 12 36 108
        valid_error_rate, runtime = self.benchmark.objective_function(config=configuration, budget=budget)
        return {'function_value': valid_error_rate,
                'cost': runtime,
                }

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration: Dict, **kwargs) -> Dict:

        test_error, runtime = self.benchmark.objective_function(config=configuration, budget=108)
        return {'function_value': test_error, 'cost': runtime}

    @staticmethod
    def get_configuration_space() -> CS.ConfigurationSpace:
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
    def get_configuration_space() -> CS.ConfigurationSpace:
        from tabular_benchmarks.nas_cifar10 import NASCifar10A
        return NASCifar10A.get_configuration_space()


class NASCifar10BBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks.nas_cifar10 import NASCifar10B
        benchmark = NASCifar10B(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10BBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)

    @staticmethod
    def get_configuration_space() -> CS.ConfigurationSpace:
        from tabular_benchmarks.nas_cifar10 import NASCifar10B
        return NASCifar10B.get_configuration_space()


class NASCifar10CBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks.nas_cifar10 import NASCifar10C
        benchmark = NASCifar10C(data_dir=str(data_path), multi_fidelity=True)
        super(NASCifar10CBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)

    @staticmethod
    def get_configuration_space() -> CS.ConfigurationSpace:
        from tabular_benchmarks.nas_cifar10 import NASCifar10C
        return NASCifar10C.get_configuration_space()


