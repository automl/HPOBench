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
0.0.4
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.3:
* Fix returned dictionary from Objective Function for ParamNetOnTime Benchmarks.
* Suppress Warning (Surrogate was created with scikit-learn version 0.18.1 and current is 0.23.2)
* Add another Search Space: Reduced.

0.0.2:
* Fix OnTime Test function:
  The `objective_test_function` of the OnTime Benchmarks now checks if the budget is the right maximum budget.
* Standardize the structure of the meta information

0.0.1:
* First implementation
"""
import warnings
import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import ParamNetDataManager

__version__ = '0.0.3'

logger = logging.getLogger('Paramnet')


class _ParamnetBase(AbstractBenchmark):

    def __init__(self, dataset: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        dataset : str
            Name for the surrogate data. Must be one of ["adult", "higgs", "letter", "mnist", "optdigits", "poker"]
        rng : np.random.RandomState, int, None
        """

        self.dataset = dataset
        self.n_epochs = 50

        super(_ParamnetBase, self).__init__(rng=rng)

        allowed_datasets = ["adult", "higgs", "letter", "mnist", "optdigits", "poker", "vehicle"]
        assert dataset in allowed_datasets, f'Requested data set is not supported. Must be one of ' \
                                            f'{", ".join(allowed_datasets)}, but was {dataset}'
        logger.info(f'Start Benchmark on dataset {dataset}')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle")
            dm = ParamNetDataManager(dataset=dataset)
            self.surrogate_objective, self.surrogate_costs = dm.load()

    @staticmethod
    def convert_config_to_array(configuration: Dict) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'ParamNet Benchmark',
                'references': ['@InProceedings{falkner-icml-18,'
                               'title     = {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},'
                               'url       = http://proceedings.mlr.press/v80/falkner18a.html'
                               'author    = {Falkner, Stefan and Klein, Aaron and Hutter, Frank}, '
                               'booktitle = {Proceedings of the 35th International Conference on Machine Learning},'
                               'pages     = {1436 - 1445},'
                               'year      = {2018}}'],
                'code': 'https://github.com/automl/HPOlib1.5/blob/development/'
                        'hpolib/benchmarks/surrogates/paramnet.py'
                }


class _ParamnetFull(_ParamnetBase):

    @staticmethod
    def convert_config_to_array(configuration: Dict) -> np.ndarray:
        """
        This function transforms a configuration to a numpy array.
        Since some of the values in the configuration space are in log space, cast it back to the original space.

        Furthermore, we round the parameters ``batch size`` and ``average unit per layer`` to their next integer.
        This is different to the original implementation of the paramnet benchmark from HPOlib1.5

        Parameters
        ----------
        configuration : Dict

        Returns
        -------
        np.ndarray - The configuration transformed back to its original space
        """
        cfg_array = np.zeros(8)
        cfg_array[0] = 10 ** configuration['initial_lr_log10']
        cfg_array[1] = round(2 ** configuration['batch_size_log2'])
        cfg_array[2] = round(2 ** configuration['average_units_per_layer_log2'])
        cfg_array[3] = 10 ** configuration['final_lr_fraction_log2']
        cfg_array[4] = configuration['shape_parameter_1']
        cfg_array[5] = configuration['num_layers']
        cfg_array[6] = configuration['dropout_0']
        cfg_array[7] = configuration['dropout_1']

        return cfg_array.reshape((1, -1))

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
            CS.UniformFloatHyperparameter('initial_lr_log10', lower=-6, upper=-2, default_value=-4, log=False),
            CS.UniformFloatHyperparameter('batch_size_log2', lower=3, upper=8, default_value=5.5, log=False),
            CS.UniformFloatHyperparameter('average_units_per_layer_log2', lower=4, upper=8, default_value=6, log=False),
            CS.UniformFloatHyperparameter('final_lr_fraction_log2', lower=-4, upper=0, default_value=-2, log=False),
            CS.UniformFloatHyperparameter('shape_parameter_1', lower=0., upper=1., default_value=0.5, log=False),
            CS.UniformIntegerHyperparameter('num_layers', lower=1, upper=5, default_value=3, log=False),
            CS.UniformFloatHyperparameter('dropout_0', lower=0., upper=0.5, default_value=0.25, log=False),
            CS.UniformFloatHyperparameter('dropout_1', lower=0., upper=0.5, default_value=0.25, log=False),
        ])
        return cs


class _ParamnetReduced(_ParamnetBase):

    @staticmethod
    def convert_config_to_array(configuration: Dict) -> np.ndarray:
        """
        This function transforms a configuration to a numpy array.
        Since some of the values in the configuration space are in log space, cast it back to the original space.

        Furthermore, we round the parameters ``batch size`` and ``average unit per layer`` to their next integer.
        This is different to the original implementation of the paramnet benchmark from HPOlib1.5

        Parameters
        ----------
        configuration : Dict

        Returns
        -------
        np.ndarray - The configuration transformed back to its original space
        """
        cfg_array = np.zeros(8)
        cfg_array[0] = 10 ** configuration['initial_lr_log10']
        cfg_array[1] = round(2 ** configuration['batch_size_log2'])
        cfg_array[2] = round(2 ** configuration['average_units_per_layer_log2'])
        cfg_array[3] = 10 ** configuration['final_lr_fraction_log2']
        cfg_array[4] = 0.5
        cfg_array[5] = configuration['num_layers']
        cfg_array[6] = configuration['dropout']
        cfg_array[7] = configuration['dropout']

        return cfg_array.reshape((1, -1))

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
            CS.UniformFloatHyperparameter('initial_lr_log10', lower=-6, upper=-2, default_value=-4, log=False),
            CS.UniformFloatHyperparameter('batch_size_log2', lower=3, upper=8, default_value=5.5, log=False),
            CS.UniformFloatHyperparameter('average_units_per_layer_log2', lower=4, upper=8, default_value=6, log=False),
            CS.UniformFloatHyperparameter('final_lr_fraction_log2', lower=-4, upper=0, default_value=-2, log=False),
            # CS.UniformFloatHyperparameter('shape_parameter_1', lower=0., upper=1., default_value=0.5, log=False),
            CS.UniformIntegerHyperparameter('num_layers', lower=1, upper=5, default_value=3, log=False),
            # CS.UniformFloatHyperparameter('dropout_0', lower=0., upper=0.5, default_value=0.25, log=False),
            CS.UniformFloatHyperparameter('dropout', lower=0., upper=0.5, default_value=0.25, log=False),
        ])
        return cs


class _ParamnetOnStepsBenchmark(_ParamnetBase):

    def __init__(self, dataset: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        dataset : str
            Name for the surrogate data. Must be one of ["adult", "higgs", "letter", "mnist", "optdigits", "poker"]
        rng : np.random.RandomState, int, None
        """
        super(_ParamnetOnStepsBenchmark, self).__init__(rng=rng, dataset=dataset)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        cfg_array = self.convert_config_to_array(configuration)
        lc = self.surrogate_objective.predict(cfg_array)[0]
        c = self.surrogate_costs.predict(cfg_array)[0]

        obj_value = lc[fidelity['step'] - 1]
        cost = (c / self.n_epochs) * fidelity['step']

        # return {'function_value': y, "cost": cost, "learning_curve": lc}
        return {'function_value': obj_value,
                "cost": cost,
                'info': {'fidelity': fidelity}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        assert fidelity['step'] == 50, f'Only querying a result for the 50. epoch is allowed, ' \
                                       f'but was {fidelity["step"]}.'
        return self.objective_function(configuration, fidelity={'step': 50}, rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the SupportVector Benchmark

        Fidelities
        ----------
        step: int - [1, 50]
            Step, when to query the surrogate model

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
            CS.UniformIntegerHyperparameter("step", lower=1, upper=50, default_value=50, log=False),
        ])
        return fidel_space

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        meta_info = _ParamnetBase.get_meta_information()
        meta_info.update({'info': 'This benchmark uses the epochs as fidelity.'})
        return meta_info


class _ParamnetOnTimeBenchmark(_ParamnetBase):

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        config_array = self.convert_config_to_array(configuration)
        lc = self.surrogate_objective.predict(config_array)[0]
        costs = self.surrogate_costs.predict(config_array)[0]

        # If we can't afford a single epoch, return 1.
        if (costs / self.n_epochs) > fidelity['budget']:
            return {'function_value': 1.0,
                    'cost': fidelity['budget'],
                    'info': {'fidelity': fidelity,
                             'learning_curve': [],
                             'observed_epochs': 0,
                             'predicted_costs': float(costs),
                             'state': 'Not enough budget'}}

        learning_curves_cost = np.linspace(costs / self.n_epochs, costs, self.n_epochs)

        if fidelity['budget'] <= costs:
            idx = np.where(learning_curves_cost <= fidelity['budget'])[0][-1]
            y = lc[idx]
            lc = lc[:idx]

        else:
            # If the budget is larger than the actual runtime, we extrapolate the learning curve
            t_left = fidelity['budget'] - costs
            n_epochs = int(t_left / (costs / self.n_epochs))
            lc = np.append(lc, np.ones(n_epochs) * lc[-1])
            y = lc[-1]

        return {'function_value': float(y),
                'cost': fidelity['budget'],
                'info': {'fidelity': fidelity,
                         'learning_curve': lc.tolist(),
                         'observed_epochs': len(lc),
                         'predicted_costs': float(costs)}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        # Get the maximum fidelity for this benchmark.
        fidelity_space = self.get_fidelity_space()
        max_budget = fidelity_space.get_hyperparameter('budget').upper

        assert fidelity['budget'] == max_budget, f'Only querying a result for a budget of {max_budget} is allowed, ' \
                                                 f'but was {fidelity["budget"]}.'
        return self.objective_function(configuration, fidelity={'budget': max_budget}, rng=rng)

    @staticmethod
    def _get_fidelity_space(dataset: str, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the SupportVector Benchmark

        Fidelities
        ----------
        budget float - [0, Max Time in S]
            Time which is used to train the network

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        budgets = {  # .   (min, max)-budget (seconds) for the different data sets
                   'adult': (9, 243),
                   'higgs': (9, 243),
                   'letter': (3, 81),
                   'mnist': (9, 243),
                   'optdigits': (1, 27),
                   'poker': (81, 2187),
        }
        min_budget, max_budget = budgets[dataset]

        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter("budget", lower=min_budget, upper=max_budget, default_value=max_budget)
        ])
        return fidel_space

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        meta_info = _ParamnetBase.get_meta_information()
        meta_info.update(
            {'note': 'This benchmark uses the training time as fidelity. '
                     'The budgets are described in I.2 Table 2 on page 17. '
                     'https://arxiv.org/pdf/1807.01774.pdf. '
                     'Also, note that the code for extrapolating the learning curve, when the budget was higher than '
                     'the total costs, was introduced in the original implemention in the HPOlib1.5',
             })
        return meta_info


class ParamNetAdultOnStepsBenchmark(_ParamnetFull, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetAdultOnStepsBenchmark, self).__init__(dataset='adult', rng=rng)


class ParamNetAdultOnTimeBenchmark(_ParamnetFull, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetAdultOnTimeBenchmark, self).__init__(dataset='adult', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='adult', seed=seed)


class ParamNetReducedAdultOnStepsBenchmark(_ParamnetReduced, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedAdultOnStepsBenchmark, self).__init__(dataset='adult', rng=rng)


class ParamNetReducedAdultOnTimeBenchmark(_ParamnetReduced, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedAdultOnTimeBenchmark, self).__init__(dataset='adult', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='adult', seed=seed)


class ParamNetHiggsOnStepsBenchmark(_ParamnetFull, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetHiggsOnStepsBenchmark, self).__init__(dataset='higgs', rng=rng)


class ParamNetHiggsOnTimeBenchmark(_ParamnetFull, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetHiggsOnTimeBenchmark, self).__init__(dataset='higgs', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='higgs', seed=seed)


class ParamNetReducedHiggsOnStepsBenchmark(_ParamnetReduced, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedHiggsOnStepsBenchmark, self).__init__(dataset='higgs', rng=rng)


class ParamNetReducedHiggsOnTimeBenchmark(_ParamnetReduced, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedHiggsOnTimeBenchmark, self).__init__(dataset='higgs', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='higgs', seed=seed)


class ParamNetLetterOnStepsBenchmark(_ParamnetFull, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetLetterOnStepsBenchmark, self).__init__(dataset='letter', rng=rng)


class ParamNetLetterOnTimeBenchmark(_ParamnetFull, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetLetterOnTimeBenchmark, self).__init__(dataset='letter', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='letter', seed=seed)


class ParamNetReducedLetterOnStepsBenchmark(_ParamnetReduced, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedLetterOnStepsBenchmark, self).__init__(dataset='letter', rng=rng)


class ParamNetReducedLetterOnTimeBenchmark(_ParamnetReduced, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedLetterOnTimeBenchmark, self).__init__(dataset='letter', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='letter', seed=seed)


class ParamNetMnistOnStepsBenchmark(_ParamnetFull, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetMnistOnStepsBenchmark, self).__init__(dataset='mnist', rng=rng)


class ParamNetMnistOnTimeBenchmark(_ParamnetFull, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetMnistOnTimeBenchmark, self).__init__(dataset='mnist', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='mnist', seed=seed)


class ParamNetReducedMnistOnStepsBenchmark(_ParamnetReduced, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedMnistOnStepsBenchmark, self).__init__(dataset='mnist', rng=rng)


class ParamNetReducedMnistOnTimeBenchmark(_ParamnetReduced, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedMnistOnTimeBenchmark, self).__init__(dataset='mnist', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='mnist', seed=seed)


class ParamNetOptdigitsOnStepsBenchmark(_ParamnetFull, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetOptdigitsOnStepsBenchmark, self).__init__(dataset='optdigits', rng=rng)


class ParamNetOptdigitsOnTimeBenchmark(_ParamnetFull, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetOptdigitsOnTimeBenchmark, self).__init__(dataset='optdigits', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='optdigits', seed=seed)


class ParamNetReducedOptdigitsOnStepsBenchmark(_ParamnetReduced, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedOptdigitsOnStepsBenchmark, self).__init__(dataset='optdigits', rng=rng)


class ParamNetReducedOptdigitsOnTimeBenchmark(_ParamnetReduced, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedOptdigitsOnTimeBenchmark, self).__init__(dataset='optdigits', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='optdigits', seed=seed)


class ParamNetPokerOnStepsBenchmark(_ParamnetFull, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetPokerOnStepsBenchmark, self).__init__(dataset='poker', rng=rng)


class ParamNetPokerOnTimeBenchmark(_ParamnetFull, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetPokerOnTimeBenchmark, self).__init__(dataset='poker', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='poker', seed=seed)


class ParamNetReducedPokerOnStepsBenchmark(_ParamnetReduced, _ParamnetOnStepsBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedPokerOnStepsBenchmark, self).__init__(dataset='poker', rng=rng)


class ParamNetReducedPokerOnTimeBenchmark(_ParamnetReduced, _ParamnetOnTimeBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        super(ParamNetReducedPokerOnTimeBenchmark, self).__init__(dataset='poker', rng=rng)

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return _ParamnetOnTimeBenchmark._get_fidelity_space(dataset='poker', seed=seed)
