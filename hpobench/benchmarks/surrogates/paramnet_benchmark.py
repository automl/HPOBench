import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import ParamNetDataManager

__version__ = '0.0.1'

logger = logging.getLogger('Paramnet')

# TODO:
#  - Vehicle
#  - Return value in case of insufficient budget
#  - get_fidelityspace is not static due to dataset
#  - what limits to use
#  - Logging.


class ParamnetBenchmark(AbstractBenchmark):

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

        super(ParamnetBenchmark, self).__init__(rng=rng)

        # TODO: Vehicle should be also allowed --> what is the time limit for it?
        allowed_datasets = ["adult", "higgs", "letter", "mnist", "optdigits", "poker"]
        assert dataset in allowed_datasets, f'Requested data set is not supported. Must be one of ' \
                                            f'{", ".join(allowed_datasets)}, but was {dataset}'

        dm = ParamNetDataManager(dataset=dataset)
        self.surrogate_objective, self.surrogate_costs = dm.load()

    @staticmethod
    def convert_config_to_array(configuration: Dict) -> np.ndarray:
        cfg_array = np.zeros(8)
        cfg_array[0] = 10 ** configuration['initial_lr_log10']
        cfg_array[1] = round(2 ** configuration['batch_size_log2'])  # todo: round
        cfg_array[2] = round(2 ** configuration['average_units_per_layer_log2'])
        cfg_array[3] = 10 ** configuration['final_lr_fraction_log2']
        cfg_array[4] = configuration['shape_parameter_1']
        cfg_array[5] = configuration['num_layers']
        cfg_array[6] = configuration['dropout_0']
        cfg_array[7] = configuration['dropout_1']

        return cfg_array.reshape((1, -1))

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function(self, configuration: Union[Dict, CS.Configuration],
                           fidelity: Union[Dict, None] = None,
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

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        assert fidelity['step'] == 50, f'Only querying a result for the 50. epoch is allowed, ' \
                                       f'but was {fidelity["step"]}.'
        return self.objective_function(configuration, fidelity={'step': 50}, rng=rng)

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
        # TODO: in origin benchmark: all uniform float and default is the mid value.
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter('initial_lr_log10', lower=-6, upper=-2, default_value=-6, log=False),
            CS.UniformFloatHyperparameter('batch_size_log2', lower=3, upper=8, default_value=3, log=False),
            CS.UniformFloatHyperparameter('average_units_per_layer_log2', lower=4, upper=8, default_value=4, log=False),
            CS.UniformFloatHyperparameter('final_lr_fraction_log2', lower=-4, upper=0, default_value=-4, log=False),
            CS.UniformFloatHyperparameter('shape_parameter_1', lower=0., upper=1., default_value=0., log=False),
            CS.UniformIntegerHyperparameter('num_layers', lower=1, upper=5, default_value=1, log=False),
            CS.UniformFloatHyperparameter('dropout_0', lower=0., upper=0.5, default_value=0., log=False),
            CS.UniformFloatHyperparameter('dropout_1', lower=0., upper=0.5, default_value=0., log=False),
        ])
        # cs.generate_all_continuous_from_bounds(SupportVectorMachine.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the SupportVector Benchmark

        Fidelities
        ----------
        step: float - [1, 50]
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
        return {'name': 'ParamNet Benchmark'}


class ParamnetTimeBenchmark(ParamnetBenchmark):
    def __init__(self, dataset: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        super(ParamnetTimeBenchmark, self).__init__(dataset, rng)

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function(self, configuration: Union[Dict, CS.Configuration],
                           fidelity: Union[Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        config_array = self.convert_config_to_array(configuration)
        lc = self.surrogate_objective.predict(config_array)[0]
        costs = self.surrogate_costs.predict(config_array)[0]

        # If we can't afford a single epoch, return TODO.
        if (costs / self.n_epochs) > fidelity['budget']:
            # TODO: Return random performance here instead
            y = 1
            return {'function_value': float(y),
                    'cost': fidelity['budget'],
                    'info': {'state': 'Not enough budget'}}

        learning_curves_cost = np.linspace(costs / self.n_epochs, costs, self.n_epochs)

        if fidelity['budget'] < costs:
            idx = np.where(learning_curves_cost < fidelity['budget'])[0][-1]
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
                'info': {'learning_curve': lc.tolist(),
                         'observed_epochs': len(lc)}}

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._check_fidelity
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, None] = None, shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        assert fidelity['step'] == 50, f'Only querying a result for the 50. epoch is allowed, ' \
                                       f'but was {fidelity["step"]}.'
        return self.objective_function(configuration, fidelity={'step': 50}, rng=rng)

    # TODO: Actually, this function takes as input the data set name. Check how to handle it, if it is send via the pyro
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
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
        budgets = {  # (min, max)-budget (seconds) for the different data sets
                   'adult': (9, 243),
                   'higgs': (9, 243),
                   'letter': (3, 81),
                   'mnist': (9, 243),
                   'optdigits': (1, 27),
                   'poker': (81, 2187),
        }
        min_budget, max_budget = budgets[self.dataset]

        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter("budget", lower=min_budget, upper=max_budget, default_value=max_budget)
        ])
        return fidel_space