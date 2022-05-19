"""
Changelog:
==========

0.0.1:
* First implementation of the Multi-Objective Fair Adult Benchmark.
"""
from typing import Union, Dict, List
import ConfigSpace as CS
import numpy as np
import logging
import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from hpobench.util.scalar import get_fitted_scaler
from hpobench.util.fairness_metrics import fairness_risk, STATISTICAL_DISPARITY, UNEQUALIZED_ODDS, UNEQUAL_OPPORTUNITY
from hpobench.util.data_manager import AdultDataManager
import time

__version__ = '0.0.1'

logger = logging.getLogger('ADULT_FAIR')


class AdultBenchmark(AbstractMultiObjectiveBenchmark):
    """
    Parameters
        ----------
        dataset : str
            One of fashion, flower.
        rng : np.random.RandomState, int, None
            Random seed for the benchmark's random state.
    """

    def __init__(self,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(AdultBenchmark, self).__init__(rng=rng)

        data_manager = AdultDataManager()
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = data_manager.load()
        self.output_class = np.unique(self.y_train)
        self.feature_names = data_manager.feature_names
        self.sensitive_feature = data_manager.sensitive_names

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'n_fc_layers', default_value=3, lower=1, upper=4, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_0', default_value=16, lower=2, upper=32, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_1', default_value=16, lower=2, upper=32, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_2', default_value=16, lower=2, upper=32, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_3', default_value=16, lower=2, upper=32, log=True
            ),
            CS.UniformFloatHyperparameter(
                'alpha', lower=10 ** -5, upper=10 ** -1, default_value=10 ** -2, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate_init', lower=10 ** -5, upper=1, default_value=10 ** -3, log=True
            ),
            CS.UniformFloatHyperparameter(
                'beta_1', lower=10 ** -3, upper=0.99, default_value=10 ** -3, log=True
            ),
            CS.UniformFloatHyperparameter(
                'beta_2', lower=10 ** -3, upper=0.99, default_value=10 ** -3, log=True
            ),
            CS.UniformFloatHyperparameter(
                'tol', lower=10 ** -5, upper=10 ** -2, default_value=10 ** -3, log=True
            ),

        ])
        return cs

    @staticmethod
    def get_objective_names(self) -> List[str]:
        return ['accuracy', 'DSP', 'DEO', 'DFP']

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:

        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'budget', lower=1, upper=200, default_value=200, log=False
            )
        ])
        print(fidelity_space)
        return fidelity_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Multi-objective Asynchronous Successive Halving',
            'references': ['@article{schmucker2021multi,'
                           'title={Multi-objective Asynchronous Successive Halving},'
                           'author={Schmucker, Robin and Donini, Michele and Zafar, Muhammad Bilal and Salinas,'
                           ' David and Archambeau, C{\'e}dric},'
                           'journal={arXiv preprint arXiv:2106.12639},'
                           'year={2021}',
                           ],
        }

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           shuffle: bool = False,
                           **kwargs) -> Dict:
        """

        Parameters
        ----------
        configuration
        fidelity: Dict, None
            epoch: int - Values: [1, 200]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)

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
            function_value : Dict
                validation_accuracy: float
                DSO: float
                DEO: float
                DFP: float
            cost : time to train the network
            info : Dict
                 valid_accuracy : float,
                 train_accuracy : float,
                 test_accuracy : float,
                 valid_cost : float,
                 test_cost : float,
                 training_cost : float,
                 elapsed_time : float,
                 DSO : float,
                 DEO : float,
                 DFP : float,
                 test_DSO : float,
                 test_DEO : float,
                 test_DFP : float,
                 fidelity : int,
        """
        self.rng = rng_helper.get_rng(rng)
        ts_start = time.time()

        budget = fidelity['budget']
        logger.debug("budget for evaluation of config:{}", budget)
        logger.debug("config for evaluation:{}", configuration)

        sensitive_rows_val = self.X_valid[:, self.feature_names.index(self.sensitive_feature)]
        sensitive_rows_test = self.X_test[:, self.feature_names.index(self.sensitive_feature)]

        # Normalize data
        scaler = get_fitted_scaler(self.X_train, "Standard")
        if scaler is not None:
            X_train = scaler(self.X_train)
            X_val = scaler(self.X_valid)
            X_test = scaler(self.X_test)

        # Create model
        hidden = [configuration['fc_layer_0'], configuration['fc_layer_1'],
                  configuration['fc_layer_2'], configuration['fc_layer_3']][:configuration['n_fc_layers']]

        configuration.pop("fc_layer_0")
        configuration.pop("fc_layer_1")
        configuration.pop("fc_layer_2")
        configuration.pop("fc_layer_3")
        configuration.pop("n_fc_layers")
        mlp = MLPClassifier(**configuration, hidden_layer_sizes=hidden, random_state=rng)

        start = time.time()
        for e in range(budget):
            mlp.partial_fit(self.X_train, self.y_train, self.output_class)
            y_pred_train = mlp.predict(X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)

        training_cost = time.time() - start

        start = time.time()
        y_pred_valid = mlp.predict(X_val)
        val_accuracy = accuracy_score(self.y_valid, y_pred_valid)
        eval_valid_runtime = time.time() - start

        start = time.time()
        y_pred_test = mlp.predict(X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        eval_test_runtime = time.time() - start

        val_statistical_disparity = fairness_risk(X_val, self.y_valid, sensitive_rows_val, mlp, STATISTICAL_DISPARITY)
        val_unequal_opportunity = fairness_risk(X_val, self.y_valid, sensitive_rows_val, mlp, UNEQUAL_OPPORTUNITY)
        val_unequalized_odds = fairness_risk(X_val, self.y_valid, sensitive_rows_val, mlp, UNEQUALIZED_ODDS)

        test_statistical_disparity = fairness_risk(X_test, self.y_test, sensitive_rows_test, mlp, STATISTICAL_DISPARITY)
        test_unequal_opportunity = fairness_risk(X_test, self.y_test, sensitive_rows_test, mlp, UNEQUAL_OPPORTUNITY)
        test_unequalized_odds = fairness_risk(X_test, self.y_test, sensitive_rows_test, mlp, UNEQUALIZED_ODDS)

        logger.debug("config:{}, val_acc:{}, test_score :{}, train score:{}, dsp:{}, deo :{}, dfp :{}",
                     configuration, val_accuracy, test_accuracy, train_accuracy, val_statistical_disparity,
                     val_unequal_opportunity,
                     val_unequalized_odds)

        elapsed_time = ts_start - time.time()

        return {'function_value': {'accuracy': val_accuracy,
                                   'DSO': val_statistical_disparity,
                                   'DEO': val_unequal_opportunity,
                                   'DFP': val_unequalized_odds
                                   },
                'cost': float(elapsed_time),
                'info': {'valid_accuracy': val_accuracy,
                         'train_accuracy': train_accuracy,
                         'test_accuracy': test_accuracy,
                         'valid_cost': eval_valid_runtime,
                         'test_cost': eval_test_runtime,
                         'training_cost': training_cost,
                         'elapsed_time': elapsed_time,
                         'DSO': val_statistical_disparity,
                         'DEO': val_unequal_opportunity,
                         'DFP': val_unequalized_odds,
                         'test_DSO': test_statistical_disparity,
                         'test_DEO': test_unequal_opportunity,
                         'test_DFP': test_unequalized_odds,
                         'fidelity': budget
                         }
                }

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                shuffle: bool = False,
                                **kwargs) -> Dict:
        """

                Parameters
                ----------
                configuration
                fidelity: Dict, None
                    epoch: int - Values: [1, 200]
                        Number of epochs an architecture was trained.
                        Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)

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
                    function_value : Dict
                        validation_accuracy: float
                        DSO: float
                        DEO: float
                        DFP: float
                    cost : time to train the network
                    info : Dict
                         valid_accuracy : float,
                         train_accuracy : float,
                         test_accuracy : float,
                         valid_cost : float,
                         test_cost : float,
                         training_cost : float,
                         elapsed_time : float,
                         DSO : float,
                         DEO : float,
                         DFP : float,
                         test_DSO : float,
                         test_DEO : float,
                         test_DFP : float,
                         fidelity : int,
        """

        # The result dict should contain already all necessary information -> Just swap the function value from valid
        # to test and the corresponding time cost
        assert fidelity['budget'] == 200, 'Only test data for the 200. epoch is available. '

        self.rng = rng_helper.get_rng(rng)

        X_train = np.vstack((self.X_train, self.X_valid))

        self.y_train = np.vstack((self.y_train[:, np.newaxis], self.y_valid[:, np.newaxis])).ravel()

        ts_start = time.time()
        budget = fidelity['budget']

        sensitive_rows = self.X_test[:, self.feature_names.index(self.sensitive_feature)]

        # Normalize data
        scaler = get_fitted_scaler(X_train, "Standard")
        if scaler is not None:
            X_train = scaler(X_train)
            X_test = scaler(self.X_test)

        # Create model
        hidden = [configuration['fc_layer_0'], configuration['fc_layer_1'],
                  configuration['fc_layer_2'], configuration['fc_layer_3']][:configuration['n_fc_layers']]

        configuration.pop("fc_layer_0")
        configuration.pop("fc_layer_1")
        configuration.pop("fc_layer_2")
        configuration.pop("fc_layer_3")
        configuration.pop("n_fc_layers")

        mlp = MLPClassifier(**configuration, hidden_layer_sizes=hidden, random_state=rng)

        start = time.time()
        for e in range(budget):
            mlp.partial_fit(X_train, self.y_train, self.output_class)
            y_pred_train = mlp.predict(X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)

        training_runtime = time.time() - start

        start = time.time()
        y_pred_valid = mlp.predict(X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred_valid)
        eval_test_runtime = time.time() - start

        test_statistical_disparity = fairness_risk(X_test, self.y_test, sensitive_rows, mlp, STATISTICAL_DISPARITY)
        test_unequal_opportunity = fairness_risk(X_test, self.y_test, sensitive_rows, mlp, UNEQUAL_OPPORTUNITY)
        test_unequalized_odds = fairness_risk(X_test, self.y_test, sensitive_rows, mlp, UNEQUALIZED_ODDS)

        elapsed_time = ts_start - time.time()

        logger.debug("config:{}, test_error:{}, test_score :{}, train score:{}, dsp:{}, deo :{}, dfp :{}",
                     configuration, test_accuracy, train_accuracy, test_statistical_disparity, test_unequal_opportunity,
                     test_unequalized_odds)

        return {'function_value': {'accuracy': test_accuracy,
                                   'DSO': test_statistical_disparity,
                                   'DEO': test_unequal_opportunity,
                                   'DFP': test_unequalized_odds
                                   },
                'cost': float(elapsed_time),
                'info': {'test_score': test_accuracy,
                         'train_score': train_accuracy,
                         'training_cost': training_runtime,
                         'test_cost': eval_test_runtime,
                         'elapsed_time': elapsed_time,
                         'DSO': test_statistical_disparity,
                         'DEO': test_unequal_opportunity,
                         'DFP': test_unequalized_odds,
                         'fidelity': budget
                         }
                }


__all__ = [AdultBenchmark]
