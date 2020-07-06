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

git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks
python setup.py install

pip install --upgrade "tensorflow>=1.12.1,<=1.15"
```
"""

from pathlib import Path
from typing import Union, Dict, Tuple

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
                           run_index: Union[int, Tuple, None] = (0, 1, 2, 3),
                           reset: bool = True,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Query the NAS-benchmark using a given configuration and a epoch (=budget).

        Parameters
        ----------
        configuration : Dict
        budget : int, None
        run_index : int, Tuple, None
            The nas benchmark has for each configuration-budget-pair results from 4 different runs.
            If multiple `run_id`s are given, the benchmark returns the mean over the given runs.
            By default all runs are used. A specific run can be chosen by setting the `run_id` to a value from [0, 3].
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
        self.rng = rng_helper.get_rng(rng)

        if isinstance(run_index, int):
            assert 0 <= run_index <= 3, f'run_index must be in [0, 3], not {run_index}'
            run_index = (run_index, )
        elif isinstance(run_index, tuple):
            assert len(run_index) != 0, 'run_index must not be empty'
            assert min(run_index) >= 0 and max(run_index) <= 3, \
                f'all run_index values must be in [0, 3], but were {run_index}'
        else:
            raise ValueError(f'run index must be one of List or Int, but was {type(run_index)}')

        if reset:
            self.reset_tracker()

        valid_rmse_list, runtime_list = [], []
        for run_id in run_index:
            valid_rmse, runtime = self.benchmark.objective_function_deterministic(config=configuration,
                                                                                  budget=budget,
                                                                                  index=run_id)
            valid_rmse_list.append(float(valid_rmse))
            runtime_list.append(float(runtime))

        valid_rmse = sum(valid_rmse_list) / len(valid_rmse_list)
        runtime = sum(runtime_list) / len(runtime_list)

        return {'function_value': float(valid_rmse),
                'cost': float(runtime),
                'info': {'valid_rmse_per_run': valid_rmse_list,
                         'runtime_per_run': runtime_list}
                }

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, configuration: Dict, rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        self.rng = rng_helper.get_rng(rng)

        test_error, runtime = self.benchmark.objective_function_test(config=configuration)

        return {'function_value': float(test_error), 'cost': float(runtime)}

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Interface to the get_configuration_space function from the FCNet Benchmark.

        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
            ConfigSpace.ConfigurationSpace - Containing the benchmark's hyperparameter
        """

        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = FCNetBenchmark.get_configuration_space()
        cs.seed(seed)
        return cs

    def reset_tracker(self):
        self.benchmark.X = []
        self.benchmark.y = []
        self.benchmark.c = []

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