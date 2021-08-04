"""
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to install the following packages besides installing the hpobench with

```pip install .[pybnn]```

Changelog:
==========
0.0.4
* Limit variance in NLL loss to prevent very large values.

0.0.3
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.2:
* Standardize the structure of the meta information
* The minimum number of burn in steps was allowed to be 0. But then Theano throws an RunTimeError. Limit the number of
  `burnin_steps` to be at least 1.

0.0.1:
* First implementation
"""
from functools import partial
import logging
import os
import time
import tempfile
from typing import Union, Dict, Any

import numpy as np
from scipy import stats

import ConfigSpace as CS

from hpobench.util.data_manager import BostonHousingData, ProteinStructureData, YearPredictionMSDData
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper

# This has to happen before any other imports
if "TMPDIR" not in os.environ:
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_name = tmpdir.name
else:
    tmpdir_name = os.environ["TMPDIR"]
os.environ["THEANO_FLAGS"] = f"base_compiledir={tmpdir_name}"

import lasagne  # noqa: E402
from sgmcmc.bnn.model import BayesianNeuralNetwork  # noqa: E402
from sgmcmc.bnn.lasagne_layers import AppendLayer  # noqa: E402


__version__ = '0.0.4'
logger = logging.getLogger('PyBnnBenchmark')


class BayesianNeuralNetworkBenchmark(AbstractBenchmark):

    def __init__(self,
                 rng: Union[np.random.RandomState, int, None] = None):
        """

        Parameters
        ----------
        rng : np.random.RandomState, int, None
        """
        super(BayesianNeuralNetworkBenchmark, self).__init__(rng=rng)

        self.n_calls = 0
        self.max_iters = 10000

        self.train, self.train_targets, self.valid, self.valid_targets, \
            self.test, self.test_targets = self.get_data()

    def get_data(self):
        raise NotImplementedError()

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a bayesian neural network with 3 layers on the defined data set and evaluates the trained model on
        the validation split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the pyBNN model
        fidelity: Dict, None
            budget : int [500 - 10000]
                number of epochs to train the model
            Fidelity parameters for the pyBNN model, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        start = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 10000))

        # See comment in objective function test
        burn_in_steps = max(1, int(configuration['burn_in'] * fidelity['budget']))

        net = partial(_get_net,
                      n_units_1=configuration['n_units_1'],
                      n_units_2=configuration['n_units_2'])

        model = BayesianNeuralNetwork(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=configuration['l_rate'],
                                      mdecay=configuration['mdecay'],
                                      burn_in=burn_in_steps,
                                      n_iters=fidelity['budget'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True,
                                      rng=self.rng)

        model.train(self.train, self.train_targets,
                    valid=self.valid, valid_targets=self.valid_targets,
                    valid_after_n_steps=100)

        mean_pred, var_pred = model.predict(self.valid)

        # Negative log-likelihood
        valid_loss = self._neg_log_likelihood(self.valid_targets, mean_pred, var_pred)

        cost = time.time() - start

        return {'function_value': float(valid_loss),
                'cost': cost,
                'info': {'fidelity': fidelity}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a bayesian neural network with 3 layers on the train and valid data split and evaluates it on the test
        split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the pyBNN model
        fidelity: Dict, None
            budget : int [500 - 10000]
                number of epochs to train the model
            Fidelity parameters for the pyBNN model, check get_fidelity_space(). Uses default (max) value if None.

            Note: The fidelity should be here the max budget (= 10000). By leaving this field empty, the maximum budget
            will be used by default.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        start = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 10000))

        # `burn_in_steps` must be at least 1, otherwise, theano will raise an RuntimeError. (Actually, the definition of
        # the config space allows as lower limit a zero. In this case, set the number of steps to 1.)
        burn_in_steps = max(1, int(configuration['burn_in'] * fidelity['budget']))

        net = partial(_get_net,
                      n_units_1=configuration['n_units_1'],
                      n_units_2=configuration['n_units_2'])

        model = BayesianNeuralNetwork(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=configuration['l_rate'],
                                      mdecay=configuration['mdecay'],
                                      burn_in=burn_in_steps,
                                      n_iters=fidelity['budget'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True,
                                      rng=self.rng)

        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))
        model.train(train, train_targets)

        mean_pred, var_pred = model.predict(self.test)
        test_loss = self._neg_log_likelihood(self.test_targets, mean_pred, var_pred)

        cost = time.time() - start
        return {'function_value': float(test_loss),
                'cost': cost,
                'info': {'fidelity': fidelity}}

    @staticmethod
    def _neg_log_likelihood(targets: np.ndarray, mean_pred: np.ndarray, var_pred: np.ndarray) -> np.ndarray:
        """ Compute the negative log likelihood for normal distributions. """
        var_pred = np.clip(var_pred, a_min=1e-10, a_max=np.inf)
        nll = [stats.norm.logpdf(targets[i], loc=mean_pred[i], scale=np.sqrt(var_pred[i]))
               for i in range(targets.shape[0])]
        return -np.mean(nll)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for the pybnn benchmark.

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
            CS.UniformFloatHyperparameter('l_rate', lower=1e-6, upper=1e-1, default_value=1e-2, log=True),
            CS.UniformFloatHyperparameter('burn_in', lower=0.0, upper=0.8, default_value=0.3),
            CS.UniformIntegerHyperparameter('n_units_1', lower=16, upper=512, default_value=64, log=True),
            CS.UniformIntegerHyperparameter('n_units_2', lower=16, upper=512, default_value=64, log=True),
            CS.UniformFloatHyperparameter('mdecay', lower=0, upper=1, default_value=0.05)
        ])

        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for pyBNN benchmark.
        Fidelities
        ----------
        budget : int : [500, 10000]
            number of epochs to train the network

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
            CS.UniformIntegerHyperparameter("budget", lower=500, upper=10000, default_value=10000, log=False)
        ])

        return fidel_space

    @staticmethod
    def get_meta_information():
        return {'name': 'BNN Benchmark',
                'references': ['@InProceedings{falkner-icml-18,'
                               'title     = {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},'
                               'url       = http://proceedings.mlr.press/v80/falkner18a.html'
                               'author    = {Falkner, Stefan and Klein, Aaron and Hutter, Frank}, '
                               'booktitle = {Proceedings of the 35th International Conference on Machine Learning},'
                               'pages     = {1436 - -1445},'
                               'year      = {2018}}'],
                'code': 'https://github.com/automl/HPOlib1.5/blob/container/hpolib/benchmarks/ml/bnn_benchmark.py'
                }


def _get_net(n_inputs: int, n_units_1: int, n_units_2: int) -> Any:
    """ Create a Network with theano """
    l_in = lasagne.layers.InputLayer(shape=(None, n_inputs))

    fc_layer_1 = lasagne.layers.DenseLayer(l_in,
                                           num_units=n_units_1,
                                           W=lasagne.init.HeNormal(),
                                           b=lasagne.init.Constant(val=0.0),
                                           nonlinearity=lasagne.nonlinearities.tanh)
    fc_layer_2 = lasagne.layers.DenseLayer(fc_layer_1,
                                           num_units=n_units_2,
                                           W=lasagne.init.HeNormal(),
                                           b=lasagne.init.Constant(val=0.0),
                                           nonlinearity=lasagne.nonlinearities.tanh)
    l_out = lasagne.layers.DenseLayer(fc_layer_2,
                                      num_units=1,
                                      W=lasagne.init.HeNormal(),
                                      b=lasagne.init.Constant(val=0.0),
                                      nonlinearity=lasagne.nonlinearities.linear)

    network = AppendLayer(l_out, num_units=1, b=lasagne.init.Constant(np.log(1e-3)))
    return network


class BNNOnToyFunction(BayesianNeuralNetworkBenchmark):
    """ Test Benchmark for the pyBNN Benchmark """
    def get_data(self):
        rng = np.random.RandomState(42)

        def toy_function(x):
            eps = rng.normal() * 0.02
            y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps
            return y

        data_x = rng.rand(1000, 1)
        data_y = np.array([toy_function(xi) for xi in data_x])[:, 0]

        train = data_x[:600]
        train_targets = data_y[:600]
        valid = data_x[600:800]
        valid_targets = data_y[600:800]
        test = data_x[800:]
        test_targets = data_y[800:]
        return train, train_targets, valid, valid_targets, test, test_targets


class BNNOnBostonHousing(BayesianNeuralNetworkBenchmark):
    def get_data(self):
        data_manager = BostonHousingData()
        return data_manager.load()


class BNNOnProteinStructure(BayesianNeuralNetworkBenchmark):
    def get_data(self):
        data_manager = ProteinStructureData()
        return data_manager.load()


class BNNOnYearPrediction(BayesianNeuralNetworkBenchmark):
    def get_data(self):
        data_manager = YearPredictionMSDData()
        return data_manager.load()
