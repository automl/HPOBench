"""
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

Prerequisites:
==============
Conda environment in which the HPOBench is installed (pip install .). Activate your environment.
```
conda activate <Name_of_Conda_HPOBench_environment>
```

1. Download data:
=================
The data will be downloaded automatically.

If you want to download the data on your own, you can download the data with the following command and then link the
hpobench-config's data-path to it.

```
wget https://www.automl.org/wp-content/uploads/2019/05/surrogates.tar.gz
```

The data consist of surrogates for different data sets. Each surrogate is a pickled scikit-learn forest. Thus, we have
a hard requirement of scikit-learn==0.23.x.


1. Clone from github:
=====================
```
git clone HPOBench
```

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .[paramnet]

```

Changelog:
==========
0.0.1:
* Initial implementation
"""

import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import SurrogateSVMDataManager

__version__ = '0.0.1'

logger = logging.getLogger('SurrogateSVM')


class SurrogateSVMBenchmark(AbstractBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        rng : np.random.RandomState, int, None
        """

        dm = SurrogateSVMDataManager()
        self.surrogate_objective, self.surrogate_costs = dm.load()

        super(SurrogateSVMBenchmark, self).__init__(rng=rng)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter('C', lower=-10, upper=10, default_value=0, log=False),
            CS.UniformFloatHyperparameter('gamma', lower=-10, upper=10, default_value=0, log=False)
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the SupportVector Benchmark.

        Following Klein et al., we set the minimum data set fraction to 1/128 of the original data set
        (N=50000 configurations).

        Fidelities
        ----------
        dataset_fraction: float - [0.1, 1]
            fraction of training data set to use

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
            CS.UniformFloatHyperparameter("dataset_fraction", lower=1/128, upper=1.0, default_value=1.0, log=False)
        ])
        return fidel_space

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'SVMOnMnist Benchmark',
                'references': ['@InProceedings{falkner-icml-18,'
                               'title       = {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},'
                               'url         = http://proceedings.mlr.press/v80/falkner18a.html'
                               'author      = {Falkner, Stefan and Klein, Aaron and Hutter, Frank}, '
                               'booktitle   = {Proceedings of the 35th International Conference on Machine Learning},'
                               'pages       = {1436 - 1445},'
                               'year        = {2018}}',
                               '@inproceedings{klein2017fast,'
                               'title       = {Fast bayesian optimization of machine learning hyperparameters on '
                               '               large datasets},'
                               'author      = {Klein, Aaron and Falkner, Stefan and Bartels, Simon and '
                               '               Hennig, Philipp and Hutter, Frank},'
                               'booktitle   = {Artificial Intelligence and Statistics},'
                               'pages       = {528--536},'
                               'year        = {2017},'
                               'organization={PMLR}}'],
                'code': 'https://github.com/automl/HPOlib1.5/blob/development/'
                        'hpolib/benchmarks/surrogates/svm.py'
                }

    @staticmethod
    def convert_config_to_array(configuration: Dict, fidelity: Dict) -> np.ndarray:
        """
        This function transforms a configuration to a numpy array.

        Parameters
        ----------
        configuration : Dict
        fidelity : Dict

        Returns
        -------
        np.ndarray - The configuration transformed back to its original space
        """
        cfg_array = np.zeros(3)
        cfg_array[0] = configuration['C']
        cfg_array[1] = configuration['gamma']
        cfg_array[2] = fidelity['dataset_fraction']
        return cfg_array.reshape((1, -1))

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        cfg_array = self.convert_config_to_array(configuration, fidelity)
        obj_value = self.surrogate_objective.predict(cfg_array)[0]
        cost = self.surrogate_costs.predict(cfg_array)[0]

        return {'function_value': obj_value,
                "cost": cost,
                'info': {'fidelity': fidelity}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        assert fidelity['dataset_fraction'] == 1, f'Only querying a result with the maximum fidelity is allowed, ' \
                                                  f'but was {fidelity["dataset_fraction"]}.'
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)
