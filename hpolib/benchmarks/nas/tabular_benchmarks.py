"""
Interface to Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search

https://github.com/automl/nas_benchmarks

How to install:
---------------

1) Download data
```
wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf fcnet_tabular_benchmarks.tar.gz
```

2) Clone + Install
```
pip install git+https://github.com/google-research/nasbench.git@master
pip install git+https//github.com/automl/nas_benchmarks.git@master

pip install --upgrade "tensorflow>=1.12.1,<=1.15"
```
"""

from pathlib import Path
from typing import Union, Dict, List

import ConfigSpace
import ConfigSpace as CS
import numpy as np
from tabular_benchmarks.fcnet_benchmark import FCNetBenchmark

import hpolib.util.rng_helper as rng_helper
from hpolib.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'


class FCNetBaseBenchmark(AbstractBenchmark):
    def __init__(self, benchmark: FCNetBenchmark,
                 data_path: Union[Path, str, None] = "./fcnet_tabular_benchmarks/",
                 rng: Union[np.random.RandomState, int, None] = None):

        super(FCNetBaseBenchmark, self).__init__(rng=rng)
        self.benchmark = benchmark
        self.data_path = data_path

    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           budget: Union[int, None] = 100,
                           run_index: Union[int, List, None] = 0,
                           reset: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:

        if isinstance(run_index, int):
            assert 0 <= run_index <= 3, f'run_index must be in [0, 3], not {run_index}'
            run_index = [run_index]
        elif isinstance(run_index, list):
            assert len(run_index) != 0, 'run_index must not be empty'
            assert min(run_index) >= 0 and max(run_index) <= 3, \
                f'all run_index values must be in [0, 3], but were {run_index}'
        else:
            raise ValueError(f'run index must be one of List or Int, but was {type(run_index)}')

        # TODO: actually do we want a new benchmark each time?
        if reset:
            self.benchmark.reset_tracker()

        self.rng = rng_helper.get_rng(rng)

        valid_rmse_list, runtime_list = [], []
        for run_id in run_index:
            valid_rmse, runtime = self.benchmark.objective_function_deterministic(config=configuration,
                                                                                  budget=budget,
                                                                                  index=run_id)
            valid_rmse_list.append(valid_rmse)
            runtime_list.append(runtime)

        valid_rmse = sum(valid_rmse_list) / len(valid_rmse_list)
        runtime = sum(runtime_list) / len(runtime_list)

        return {'function_value': valid_rmse,
                'cost': runtime,
                'info': {'valid_rmse_per_run': valid_rmse_list,
                         'runtime_per_run': runtime_list}
                }

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration: Dict, **kwargs) -> Dict:

        test_error, runtime = self.benchmark.objective_function_test(config=configuration)

        return {'function_value': test_error, 'cost': runtime}

    @staticmethod
    def get_configuration_space() -> CS.ConfigurationSpace:
        return FCNetBenchmark.get_configuration_space()

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search',
                'references': ['Aaron Klein, Frank Hutter',
                               'Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization',
                               'https://arxiv.org/abs/1905.04970',
                               'https://github.com/automl/nas_benchmarks'],
                }


class SliceLocalizationBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks import FCNetSliceLocalizationBenchmark
        benchmark = FCNetSliceLocalizationBenchmark(data_dir=str(data_path))
        super(SliceLocalizationBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)


class ProteinStructureBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks import FCNetProteinStructureBenchmark
        benchmark = FCNetProteinStructureBenchmark(data_dir=str(data_path))
        super(ProteinStructureBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)


class NavalPropulsionBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks import FCNetNavalPropulsionBenchmark
        benchmark = FCNetNavalPropulsionBenchmark(data_dir=str(data_path))
        super(NavalPropulsionBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)


class ParkinsonsTelemonitoringBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/'):
        from tabular_benchmarks import FCNetParkinsonsTelemonitoringBenchmark
        benchmark = FCNetParkinsonsTelemonitoringBenchmark(data_dir=str(data_path))
        super(ParkinsonsTelemonitoringBenchmark, self).__init__(benchmark=benchmark, data_path=data_path)