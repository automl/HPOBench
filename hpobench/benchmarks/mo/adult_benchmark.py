"""
Changelog:
==========

0.0.1:
* First implementation of the Multi-Objective Fair Adult Benchmark.
"""
import logging
import time
from typing import Union, Dict, List, Any, Tuple

import ConfigSpace as CS
import numpy as np
from ConfigSpace.conditions import GreaterThanCondition
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark
from hpobench.dependencies.mo.fairness_metrics import fairness_risk, STATISTICAL_DISPARITY, UNEQUALIZED_ODDS, \
    UNEQUAL_OPPORTUNITY
from hpobench.dependencies.mo.scalar import get_fitted_scaler
from hpobench.util.data_manager import AdultDataManager

__version__ = '0.0.1'

logger = logging.getLogger('ADULT_FAIR')


class AdultBenchmark(AbstractMultiObjectiveBenchmark):
    def __init__(self,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """
        Multi-objective fairness HPO task. Optimize the HP of a NN on the adult data set.

        Parameters
        ----------
        rng : np.random.RandomState, int, None
            Random seed for the benchmark's random state.
        """
        super(AdultBenchmark, self).__init__(rng=rng, **kwargs)

        data_manager = AdultDataManager()
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = data_manager.load()
        self.output_class = np.unique(self.y_train)
        self.feature_names = data_manager.feature_names
        self.sensitive_feature = data_manager.sensitive_names

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for the MLP.

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_fc_layers', default_value=3, lower=1, upper=4, log=False),
            CS.UniformIntegerHyperparameter('fc_layer_0', default_value=16, lower=2, upper=32, log=True),
            CS.UniformIntegerHyperparameter('fc_layer_1', default_value=16, lower=2, upper=32, log=True),
            CS.UniformIntegerHyperparameter('fc_layer_2', default_value=16, lower=2, upper=32, log=True),
            CS.UniformIntegerHyperparameter('fc_layer_3', default_value=16, lower=2, upper=32, log=True),
            CS.UniformFloatHyperparameter('alpha', lower=10**-5, upper=10**-1, default_value=10**-2, log=True),
            CS.UniformFloatHyperparameter('learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True),
            CS.UniformFloatHyperparameter('beta_1', lower=10**-3, upper=0.99, default_value=10**-3, log=True),
            CS.UniformFloatHyperparameter('beta_2', lower=10**-3, upper=0.99, default_value=10**-3, log=True),
            CS.UniformFloatHyperparameter('tol', lower=10**-5, upper=10**-2, default_value=10**-3, log=True),
        ])

        cs.add_conditions([
            # Add the fc_layer_1 (2nd layer) if we allow more than 1 `n_fc_layers`, and so on...
            GreaterThanCondition(cs.get_hyperparameter('fc_layer_1'), cs.get_hyperparameter('n_fc_layers'), 1),
            GreaterThanCondition(cs.get_hyperparameter('fc_layer_2'), cs.get_hyperparameter('n_fc_layers'), 2),
            GreaterThanCondition(cs.get_hyperparameter('fc_layer_3'), cs.get_hyperparameter('n_fc_layers'), 3),
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters.

        Fidelities
        ----------
        budget: int - Values: [1, 200]
            Number of epochs an architecture was trained.
            Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'budget', lower=1, upper=200, default_value=200, log=False
            )
        )
        return fidelity_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Multi-objective Asynchronous Successive Halving',
            'references':
                ['@article{schmucker2021multi,'
                 'title={Multi-objective Asynchronous Successive Halving},'
                 'author={Schmucker, Robin and Donini, Michele and Zafar, Muhammad Bilal and Salinas,'
                 ' David and Archambeau, C{\'e}dric},'
                 'journal={arXiv preprint arXiv:2106.12639},'
                 'year={2021}']}

    @staticmethod
    def get_objective_names() -> List[str]:
        """Get a list of objectives evaluated in the objective_function. """
        return ['accuracy', 'DSP', 'DEO', 'DFP']

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           shuffle: bool = False,
                           **kwargs) -> Dict:
        """
        Objective function for the multi-objective adult benchmark.

        We train a NN and evaluate its performance using fairness metrics.
        This function returns the performance on the validation set.
        However, we report also train and test performance.

        Parameters
        ----------
        configuration: Dict, CS.Configuration
            Configuration for the MLP model.
        fidelity: Dict, None
            budget: int - Values: [1, 200]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        shuffle: bool, None
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        kwargs

        Returns
        -------
        Dict -
            function_value : Dict - validation metrics after training on train
                accuracy: float
                DSO: float
                DEO: float
                DFP: float
            cost : time to train the network
            info : Dict
                 train_accuracy : float
                 valid_accuracy : float
                 test_accuracy : float
                 training_cost : float - time to train the network. see `training_cost`
                 total_cost : float - elapsed time for the entire obj_func call,
                 eval_train_cost : float - time to compute metrics on training split
                 eval_valid_cost : float - time to compute metrics on validation split
                 eval_test_cost : float - time to compute metrics on test split
                 train_DSO : float
                 train_DEO : float
                 train_DFP : float
                 valid_DSO : float
                 valid_DEO : float
                 valid_DFP : float
                 test_DSO : float
                 test_DEO : float
                 test_DFP : float
                 fidelity : int
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        if shuffle:
            self._shuffle_data(rng=self.rng, shuffle_valid=False)

        ts_start = time.time()

        budget = fidelity['budget']
        logger.debug(f"budget for evaluation of config:{budget}")
        logger.debug(f"config for evaluation:{configuration}")

        sensitive_rows_train = self.X_train[:, self.feature_names.index(self.sensitive_feature)]
        sensitive_rows_val = self.X_valid[:, self.feature_names.index(self.sensitive_feature)]
        sensitive_rows_test = self.X_test[:, self.feature_names.index(self.sensitive_feature)]

        X_train, X_valid, X_test = self.X_train.copy(), self.X_valid.copy(), self.X_test.copy()

        # Normalize data
        scaler = get_fitted_scaler(X_train, "Standard")
        if scaler is not None:
            X_train = scaler(X_train)
            X_valid = scaler(X_valid)
            X_test = scaler(X_test)

        # Create model. The parameters fc_layer_1-3 might not be included in the search space.
        hidden = [configuration['fc_layer_0'],
                  configuration.get('fc_layer_1', None),
                  configuration.get('fc_layer_2', None),
                  configuration.get('fc_layer_3', None)][:configuration['n_fc_layers']]

        for item in ['fc_layer_0', 'fc_layer_1', 'fc_layer_2', 'fc_layer_3', 'n_fc_layers']:
            if item in configuration:
                configuration.pop(item)

        # We deviate here from the original implementation. They have called `budget`-times mlp.partial_fit().
        # We call `.fit()` due to efficiency aspects.
        mlp = MLPClassifier(**configuration, hidden_layer_sizes=hidden, shuffle=shuffle,
                            random_state=self.rng, max_iter=budget)

        mlp.fit(X_train, self.y_train)
        training_cost = time.time() - ts_start

        train_accuracy, train_statistical_disparity, train_unequal_opportunity, train_unequalized_odds, \
            eval_train_runtime = \
            AdultBenchmark._compute_metrics_on_split(X_train, self.y_train, sensitive_rows_train, mlp)

        val_accuracy, val_statistical_disparity, val_unequal_opportunity, val_unequalized_odds, eval_valid_runtime = \
            AdultBenchmark._compute_metrics_on_split(X_valid, self.y_valid, sensitive_rows_val, mlp)

        test_accuracy, test_statistical_disparity, test_unequal_opportunity, test_unequalized_odds, eval_test_runtime =\
            AdultBenchmark._compute_metrics_on_split(X_test, self.y_test, sensitive_rows_test, mlp)

        logger.debug(f"config: {configuration}, val_acc: {val_accuracy}, test_score: {test_accuracy}, "
                     f"train score: {train_accuracy}, dsp: {val_statistical_disparity}, "
                     f"deo :{val_unequal_opportunity}, dfp :{val_unequalized_odds}")

        elapsed_time = time.time() - ts_start

        return {'function_value': {'accuracy': float(val_accuracy),
                                   'DSO': float(val_statistical_disparity),
                                   'DEO': float(val_unequal_opportunity),
                                   'DFP': float(val_unequalized_odds)
                                   },
                'cost': training_cost,
                'info': {'train_accuracy': float(train_accuracy),
                         'valid_accuracy': float(val_accuracy),
                         'test_accuracy': float(test_accuracy),
                         'training_cost': training_cost,
                         'total_cost': elapsed_time,
                         'eval_train_cost': eval_train_runtime,
                         'eval_valid_cost': eval_valid_runtime,
                         'eval_test_cost': eval_test_runtime,
                         'train_DSO': float(train_statistical_disparity),
                         'train_DEO': float(train_unequal_opportunity),
                         'train_DFP': float(train_unequalized_odds),
                         'valid_DSO': float(val_statistical_disparity),
                         'valid_DEO': float(val_unequal_opportunity),
                         'valid_DFP': float(val_unequalized_odds),
                         'test_DSO': float(test_statistical_disparity),
                         'test_DEO': float(test_unequal_opportunity),
                         'test_DFP': float(test_unequalized_odds),
                         'fidelity': budget
                         }
                }

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                shuffle: Union[bool, None] = False,
                                **kwargs) -> Dict:
        """
        Objective function for the multi-objective adult benchmark.

        We train a NN and evaluate its performance using fairness metrics.
        This function returns the performance on the test set.

        Parameters
        ----------
        configuration: Dict, CS.Configuration
            Configuration for the MLP model.
            Use default configuration if None.
        fidelity: Dict, CS.Configuration, None
            epoch: int - Values: [1, 200]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        shuffle: bool, None
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        kwargs

        Returns
        -------
        Dict -
            function_value : Dict - test metrics reported after training on (train+valid)
                accuracy: float
                DSO: float
                DEO: float
                DFP: float
            cost : float - time to train the network. see `training_cost`
            info : Dict
                 train_accuracy : float
                 test_accuracy : float
                 training_cost : float
                 total_cost : float - elapsed time for the entire obj_func_test call,
                 eval_train_cost : float - time to compute metrics on training split
                 eval_test_cost : float - time to compute metrics on test split
                 train_DSO : float
                 train_DEO : float
                 train_DFP : float
                 test_DSO : float
                 test_DEO : float
                 test_DFP : float
                 fidelity : int
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if shuffle:
            self._shuffle_data(self.rng, shuffle_valid=True)

        ts_start = time.time()

        budget = fidelity['budget']

        X_train, X_valid, X_test = self.X_train.copy(), self.X_valid.copy(), self.X_test.copy()
        X_train = np.vstack((X_train, X_valid))
        y_train = np.vstack((self.y_train[:, np.newaxis], self.y_valid[:, np.newaxis])).ravel()

        sensitive_rows_train = X_train[:, self.feature_names.index(self.sensitive_feature)]
        sensitive_rows_test = X_test[:, self.feature_names.index(self.sensitive_feature)]

        # Normalize data
        scaler = get_fitted_scaler(X_train, "Standard")
        if scaler is not None:
            X_train = scaler(X_train)
            X_test = scaler(X_test)

        # Create model. The parameters fc_layer_1-3 might not be included in the search space.
        hidden = [configuration['fc_layer_0'],
                  configuration.get('fc_layer_1', None),
                  configuration.get('fc_layer_2', None),
                  configuration.get('fc_layer_3', None)][:configuration['n_fc_layers']]

        for item in ['fc_layer_0', 'fc_layer_1', 'fc_layer_2', 'fc_layer_3', 'n_fc_layers']:
            if item in configuration:
                configuration.pop(item)

        # We deviate here from the original implementation. They have called `budget`-times mlp.partial_fit().
        # We call `.fit()` due to efficiency aspects.
        mlp = MLPClassifier(**configuration, hidden_layer_sizes=hidden, shuffle=shuffle,
                            random_state=rng, max_iter=budget)
        mlp.fit(X_train, y_train)
        training_cost = time.time() - ts_start

        train_accuracy, train_statistical_disparity, train_unequal_opportunity, train_unequalized_odds, \
            eval_train_runtime = \
            AdultBenchmark._compute_metrics_on_split(X_train, y_train, sensitive_rows_train, mlp)

        test_accuracy, test_statistical_disparity, test_unequal_opportunity, test_unequalized_odds, eval_test_runtime =\
            AdultBenchmark._compute_metrics_on_split(X_test, self.y_test, sensitive_rows_test, mlp)

        elapsed_time = time.time() - ts_start

        logger.debug(f"config:{configuration}, test_score: {test_accuracy}, train score:{train_accuracy},"
                     f"dsp:{test_statistical_disparity}, deo :{test_unequal_opportunity}, dfp :{test_unequalized_odds}")

        return {'function_value': {'accuracy': float(test_accuracy),
                                   'DSO': float(test_statistical_disparity),
                                   'DEO': float(test_unequal_opportunity),
                                   'DFP': float(test_unequalized_odds)
                                   },
                'cost': training_cost,
                'info': {'train_accuracy': float(train_accuracy),
                         'test_accuracy': float(test_accuracy),
                         'training_cost': training_cost,
                         'total_cost': elapsed_time,
                         'eval_train_cost': eval_train_runtime,
                         'eval_test_cost': eval_test_runtime,
                         'train_DSO': float(train_statistical_disparity),
                         'train_DEO': float(train_unequal_opportunity),
                         'train_DFP': float(train_unequalized_odds),
                         'test_DSO': float(test_statistical_disparity),
                         'test_DEO': float(test_unequal_opportunity),
                         'test_DFP': float(test_unequalized_odds),
                         'fidelity': budget
                         }
                }

    @staticmethod
    def _compute_metrics_on_split(
            x_split: np.ndarray, y_split: np.ndarray, sensitive_rows: Any,  mlp: Any
    ) -> Tuple:

        start = time.time()
        _y_pred = mlp.predict(x_split)
        accuracy = accuracy_score(y_split, _y_pred)
        statistical_disparity = fairness_risk(x_split, y_split, sensitive_rows, mlp, STATISTICAL_DISPARITY)
        unequal_opportunity = fairness_risk(x_split, y_split, sensitive_rows, mlp, UNEQUAL_OPPORTUNITY)
        unequalized_odds = fairness_risk(x_split, y_split, sensitive_rows, mlp, UNEQUALIZED_ODDS)
        runtime = time.time() - start
        return accuracy, statistical_disparity, unequal_opportunity, unequalized_odds, runtime

    def _shuffle_data(self, rng=None, shuffle_valid=False) -> None:
        """
        Reshuffle the training data.

        Parameters
        ----------
        rng
            If 'rng' is None, the training idx are shuffled according to the class-random-state
        shuffle_valid: bool, None
            If true, shuffle the validation data. Defaults to False.
        """
        random_state = rng_helper.get_rng(rng, self.rng)

        train_idx = np.arange(len(self.X_train))
        random_state.shuffle(train_idx)
        self.X_train = self.X_train[train_idx]
        self.y_train = self.y_train[train_idx]

        if shuffle_valid:
            valid_idx = np.arange(len(self.X_valid))
            random_state.shuffle(valid_idx)
            self.X_valid = self.X_valid[valid_idx]
            self.y_valid = self.y_valid[valid_idx]


__all__ = ['AdultBenchmark']
