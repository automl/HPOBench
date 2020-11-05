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

"""

from pathlib import Path
from typing import Union, Dict, Any

import ConfigSpace as CS
import numpy as np
from tabular_benchmarks.nas_cifar10 import NASCifar10

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'

MAX_EDGES = 9
VERTICES = 7


class NASCifar10BaseBenchmark(AbstractBenchmark):
    def __init__(self, benchmark: NASCifar10, data_path: Union[Path, str, None] = "./",
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
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

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Query the NAS-benchmark using a given configuration and a epoch (=budget).

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
                fidelity : used fidelities in this evaluation
        """
        self.benchmark.reset_tracker()

        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        # Returns (valid_error_rate: 1, runtime: 0) if it is invalid, e.g. config not valid or
        # budget not in 4 12 36 108
        valid_error_rate, runtime = self.benchmark.objective_function(config=configuration,
                                                                      budget=fidelity["budget"])
        return {'function_value': float(valid_error_rate),
                'cost': float(runtime),
                'info': {'fidelity': fidelity}
                }

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, None] = None,
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
        return self.objective_function(configuration=configuration, fidelity=fidelity, rng=rng,
                                       **kwargs)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
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


class NASCifar10ABenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/',
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
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


class NASCifar10BBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/',
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
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


class NASCifar10CBenchmark(NASCifar10BaseBenchmark):
    def __init__(self, data_path: Union[Path, str, None] = './fcnet_tabular_benchmarks/',
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
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
