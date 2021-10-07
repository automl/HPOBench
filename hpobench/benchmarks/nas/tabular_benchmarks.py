"""
Interface to Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search

https://github.com/automl/nas_benchmarks
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

1. Download data
================
```
wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf fcnet_tabular_benchmarks.tar.gz
```

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .[tabular_benchmarks]
pip install git+https://github.com/google-research/nasbench.git@master

git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks
python setup.py install
```

Changelog:
==========
0.0.5
* Add for each benchmark a new one with a different fidelity space.
  The new fidelity space corresponds to the fidelity space in the DEHB paper.

0.0.4
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.3:
* Standardize the structure of the meta information

0.0.2:
* The objective function takes as input now the parameter run_index. Allowed values are Tuple(0-3), 0, 1, 2, 3, None.
  This value specifies which seeds are used. The user can specify a single index or a tuple with indices.
  If the user wants to use a randomly drawn run_index, they can simply set the value explicitly to None.

0.0.1:
* First implementation
"""
import logging

from pathlib import Path
from typing import Union, Dict, Tuple, List

import ConfigSpace as CS
import numpy as np
from tabular_benchmarks.fcnet_benchmark import FCNetBenchmark

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.5'
logger = logging.getLogger('TabularBenchmark')


class FCNetBaseBenchmark(AbstractBenchmark):
    def __init__(self, benchmark: FCNetBenchmark,
                 data_path: Union[Path, str, None] = "./fcnet_tabular_benchmarks/",
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        super(FCNetBaseBenchmark, self).__init__(rng=rng)
        self.benchmark = benchmark
        self.data_path = data_path

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           run_index: Union[int, Tuple, None] = (0, 1, 2, 3),
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
            The nas benchmark has for each configuration-budget-pair results from 4 different runs.
            If multiple `run_id`s are given, the benchmark returns the mean over the given runs.
            By default (no parameter is specified) all runs are used. A specific run can be chosen by setting the
            `run_id` to a value from [0, 3].
            When this value is explicitly set to `None`, the function will use a random seed.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is
            possible to pass a parameter ``rng`` as 'int' or 'np.random.RandomState' to this
            function. If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict with valid_rmse_per_run, runtime_per_run
            info : Dict
                valid_rmse_per_run
                runtime_per_run
                fidelity : used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng)

        if isinstance(run_index, int):
            assert 0 <= run_index <= 3, f'run_index must be in [0, 3], not {run_index}'
            run_index = (run_index, )
        elif isinstance(run_index, (Tuple, List)):
            assert len(run_index) != 0, 'run_index must not be empty'
            if len(set(run_index)) != len(run_index):
                logger.debug('There are some values more than once in the run_index. We remove the redundant entries.')
            run_index = tuple(set(run_index))
            assert min(run_index) >= 0 and max(run_index) <= 3, \
                f'all run_index values must be in [0, 3], but were {run_index}'
        elif run_index is None:
            logger.debug('The run index is explicitly set to None! A random seed will be selected.')
            run_index = tuple(self.rng.choice((0, 1, 2, 3), size=1))
        else:
            raise ValueError(f'run index must be one of Tuple or Int, but was {type(run_index)}')

        self._reset_tracker()

        valid_rmse_list, runtime_list = [], []
        for run_id in run_index:
            valid_rmse, runtime = self.benchmark.objective_function_deterministic(config=configuration,
                                                                                  budget=fidelity["budget"],
                                                                                  index=run_id)
            valid_rmse_list.append(float(valid_rmse))
            runtime_list.append(float(runtime))

        valid_rmse = sum(valid_rmse_list) / len(valid_rmse_list)
        runtime = sum(runtime_list)

        return {'function_value': float(valid_rmse),
                'cost': float(runtime),
                'info': {'valid_rmse_per_run': valid_rmse_list,
                         'runtime_per_run': runtime_list,
                         'fidelity': fidelity},
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """

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
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                valid_rmse_per_run
                runtime_per_run
                fidelity : used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        default_fidelity = self.get_fidelity_space().get_default_configuration().get_dictionary()
        assert fidelity == default_fidelity, 'Test function works only on the highest budget.'
        result = self.benchmark.objective_function_test(configuration)

        return {'function_value': float(result[0]),
                'cost': float(result[1]),
                'info': {'fidelity': fidelity},
                }

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
        CS.ConfigurationSpace -
            Containing the benchmark's hyperparameter
        """

        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = FCNetBenchmark.get_configuration_space()
        cs.seed(seed)
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the FCNetBaseBenchmark

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
            CS.UniformIntegerHyperparameter('budget', lower=1, upper=100, default_value=100)
        ])

        return fidel_space

    def _reset_tracker(self):
        """ Helper function to reset the internal memory of the benchmark. """
        self.benchmark.X = []
        self.benchmark.y = []
        self.benchmark.c = []

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
                               'https://github.com/automl/nas_benchmarks'],
                'code': 'https://github.com/automl/nas_benchmarks',
                }


class SliceLocalizationBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetSliceLocalizationBenchmark

        benchmark = FCNetSliceLocalizationBenchmark(data_dir=str(data_path))
        super(SliceLocalizationBenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng, **kwargs)


class ProteinStructureBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetProteinStructureBenchmark
        benchmark = FCNetProteinStructureBenchmark(data_dir=str(data_path))
        super(ProteinStructureBenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng, **kwargs)


class NavalPropulsionBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetNavalPropulsionBenchmark
        benchmark = FCNetNavalPropulsionBenchmark(data_dir=str(data_path))
        super(NavalPropulsionBenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng, **kwargs)


class ParkinsonsTelemonitoringBenchmark(FCNetBaseBenchmark):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetParkinsonsTelemonitoringBenchmark
        benchmark = FCNetParkinsonsTelemonitoringBenchmark(data_dir=str(data_path))
        super(ParkinsonsTelemonitoringBenchmark, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng,
                                                                **kwargs)


class _FCNetBaseBenchmarkOriginal(FCNetBaseBenchmark):

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        This fidelity space differs from the one above in its lower bound.
        The benchmark above enables the user to access the entire dataset, while this one reproduces the
        experiments from DEHB
        [DEHB](https://github.com/automl/DEHB/tree/937dd5cf48e79f6d587ea2ff408cb5ad9a8dce46/dehb/examples)

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
            CS.UniformIntegerHyperparameter('budget', lower=3, upper=100, default_value=100)
        ])

        return fidel_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        meta_information = FCNetBaseBenchmark.get_meta_information()
        meta_information['note'] = \
            'This version of the benchmark implements the fidelity space defined in the DEHB paper. ' \
            'See [DEHB](https://github.com/automl/DEHB/tree/937dd5cf48e79f6d587ea2ff408cb5ad9a8dce46/dehb/examples)'
        return meta_information


class SliceLocalizationBenchmarkOriginal(_FCNetBaseBenchmarkOriginal):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetSliceLocalizationBenchmark

        benchmark = FCNetSliceLocalizationBenchmark(data_dir=str(data_path))
        super(SliceLocalizationBenchmarkOriginal, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng,
                                                                 **kwargs)


class ProteinStructureBenchmarkOriginal(_FCNetBaseBenchmarkOriginal):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetProteinStructureBenchmark
        benchmark = FCNetProteinStructureBenchmark(data_dir=str(data_path))
        super(ProteinStructureBenchmarkOriginal, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng,
                                                                **kwargs)


class NavalPropulsionBenchmarkOriginal(_FCNetBaseBenchmarkOriginal):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetNavalPropulsionBenchmark
        benchmark = FCNetNavalPropulsionBenchmark(data_dir=str(data_path))
        super(NavalPropulsionBenchmarkOriginal, self).__init__(benchmark=benchmark, data_path=data_path, rng=rng,
                                                               **kwargs)


class ParkinsonsTelemonitoringBenchmarkOriginal(_FCNetBaseBenchmarkOriginal):

    def __init__(self, data_path: Union[Path, str, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        from hpobench import config_file
        data_path = Path(data_path) if data_path is not None else config_file.data_dir / 'fcnet_tabular_benchmarks'

        from tabular_benchmarks import FCNetParkinsonsTelemonitoringBenchmark
        benchmark = FCNetParkinsonsTelemonitoringBenchmark(data_dir=str(data_path))
        super(ParkinsonsTelemonitoringBenchmarkOriginal, self).__init__(benchmark=benchmark, data_path=data_path,
                                                                        rng=rng, **kwargs)


__all__ = ["SliceLocalizationBenchmark", "SliceLocalizationBenchmarkOriginal",
           "ProteinStructureBenchmark", "ProteinStructureBenchmarkOriginal",
           "NavalPropulsionBenchmark", "NavalPropulsionBenchmarkOriginal",
           "ParkinsonsTelemonitoringBenchmark", "ParkinsonsTelemonitoringBenchmarkOriginal"]
