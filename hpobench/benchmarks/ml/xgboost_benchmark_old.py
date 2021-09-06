"""

Changelog:
==========
0.0.3
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.2:
* Change the search space definiton to match the paper: (https://arxiv.org/pdf/1802.09596.pdf)
    eta:                [1e-5, 1] (def: 0.3)    ->  [2**-10, 1] (def: 0.3)
    min_child_weight:   [0,05, 10] (def: 1)     ->  [1, 2**7] (def: 1)
    colsample_bytree:   [0,05, 1] (def: 1)      ->  [0.01, 1] (def: 1)
    colsample_bylevel:  [0,05, 1] (def: 1)      ->  [0.01, 1] (def: 1)
    reg_lambda:         [1e-5, 2] (def: 1)      ->  [2**-10, 2**10] (def: 1)
    reg_alpha:          [1e-5, 2] (def: 1e-5)   ->  [2**-10, 2**10] (def: 1)
    max_depth:          -                       ->  [1, 15] (def: 6)
    subsample_per_it:   -                       ->  [0.01, 1] (def: 1)
    [booster:            -                      ->  [gbtree, gblinear, dart] (def: gbtree)]  *)

    *) This parameter is only in the XGBoostExtendedBenchmark. Not in the XGBoostBenchmark class.

* Increase the fidelity `n_estimators`
    n_estimators        [2, 128] (def: 128)     ->  [1, 256] (def: 256)

* Add class to optimize also the used booster method: (gbtree, gblinear or dart)
    We have introduced a new class, which adds the used booster as parameter to the configuration space. To read more
    about booster, please take a look in the official XGBoost-documentation (https://xgboost.readthedocs.io/en/latest).


0.0.1:
* First implementation of a XGBoost Benchmark.


"""

import logging
import time
from typing import Union, Tuple, Dict, List

import ConfigSpace as CS
import numpy as np
import xgboost as xgb
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.openml_data_manager import OpenMLHoldoutDataManager

__version__ = '0.0.3'

logger = logging.getLogger('XGBBenchmark')


class XGBoostBenchmark(AbstractBenchmark):

    def __init__(self, task_id: Union[int, None] = None, n_threads: int = 1,
                 rng: Union[np.random.RandomState, int, None] = None):
        """

        Parameters
        ----------
        task_id : int, None
        n_threads  : int, None
        rng : np.random.RandomState, int, None
        """

        super(XGBoostBenchmark, self).__init__(rng=rng)
        self.n_threads = n_threads
        self.task_id = task_id
        self.accuracy_scorer = make_scorer(accuracy_score)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, variable_types = \
            self.get_data()
        self.categorical_data = np.array([var_type == 'categorical' for var_type in variable_types])

        # XGB needs sorted data. Data should be (Categorical + numerical) not mixed.
        categorical_idx = np.argwhere(self.categorical_data)
        continuous_idx = np.argwhere(~self.categorical_data)
        sorting = np.concatenate([categorical_idx, continuous_idx]).squeeze()
        self.categorical_data = self.categorical_data[sorting]
        self.x_train = self.x_train[:, sorting]
        self.x_valid = self.x_valid[:, sorting]
        self.x_test = self.x_test[:, sorting]

        nan_columns = np.all(np.isnan(self.x_train), axis=0)
        self.categorical_data = self.categorical_data[~nan_columns]

        self.x_train, self.x_valid, self.x_test, self.categories = \
            OpenMLHoldoutDataManager.replace_nans_in_cat_columns(self.x_train, self.x_valid, self.x_test,
                                                                 is_categorical=self.categorical_data)

        # Determine the number of categories in the labels.
        # In case of binary classification ``self.num_class`` has to be 1 for xgboost.
        self.num_class = len(np.unique(np.concatenate([self.y_train, self.y_test, self.y_valid])))
        self.num_class = 1 if self.num_class == 2 else self.num_class

        self.train_idx = self.rng.choice(a=np.arange(len(self.x_train)),
                                         size=len(self.x_train),
                                         replace=False)

        # Similar to [Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets]
        # (https://arxiv.org/pdf/1605.07079.pdf),
        # use 10 time the number of classes as lower bound for the dataset fraction
        n_classes = np.unique(self.y_train).shape[0]
        self.lower_bound_train_size = (10 * n_classes) / self.x_train.shape[0]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """ Loads the data given a task or another source. """

        assert self.task_id is not None, NotImplementedError('No task-id given. Please either specify a task-id or '
                                                             'overwrite the get_data method.')

        data_manager = OpenMLHoldoutDataManager(openml_task_id=self.task_id, rng=self.rng)
        x_train, y_train, x_val, y_val, x_test, y_test = data_manager.load()

        return x_train, y_train, x_val, y_val, x_test, y_test, data_manager.variable_types

    def shuffle_data(self, rng=None):
        """ Reshuffle the training data. If 'rng' is None, the training idx are shuffled according to the
        class-random-state"""
        random_state = rng_helper.get_rng(rng, self.rng)
        random_state.shuffle(self.train_idx)

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a XGBoost model given a hyperparameter configuration and
        evaluates the model on the validation set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the XGBoost model
        fidelity: Dict, None
            Fidelity parameters for the XGBoost model, check get_fidelity_space(). Uses default (max) value if None.
        shuffle : bool
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
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
                train_loss : trainings loss
                fidelity : used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if shuffle:
            self.shuffle_data(self.rng)

        start = time.time()

        if self.lower_bound_train_size > fidelity['dataset_fraction']:
            train_data_fraction = self.lower_bound_train_size
            logger.warning(f'The given data set fraction is lower than the lower bound (10 * number of classes.) '
                           f'Increase the fidelity from {fidelity["dataset_fraction"]:.8f} to '
                           f'{self.lower_bound_train_size:.8f}')
        else:
            train_data_fraction = fidelity['dataset_fraction']

        train_idx = self.train_idx[:int(len(self.train_idx) * train_data_fraction)]

        model = self._get_pipeline(n_estimators=fidelity["n_estimators"], **configuration)
        model.fit(X=self.x_train[train_idx], y=self.y_train[train_idx])

        train_loss = 1 - self.accuracy_scorer(model, self.x_train[train_idx], self.y_train[train_idx])
        val_loss = 1 - self.accuracy_scorer(model, self.x_valid, self.y_valid)
        cost = time.time() - start

        return {'function_value': float(val_loss),
                'cost': cost,
                'info': {'train_loss': float(train_loss),
                         'fidelity': fidelity}
                }

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a XGBoost model with a given configuration on both the train
        and validation data set and evaluates the model on the test data set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the XGBoost Model
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        shuffle : bool
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : test loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        default_dataset_fraction = self.get_fidelity_space().get_hyperparameter('dataset_fraction').default_value
        if fidelity['dataset_fraction'] != default_dataset_fraction:
            raise NotImplementedError(f'Test error can not be computed for dataset_fraction <= '
                                      f'{default_dataset_fraction}')

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if shuffle:
            self.shuffle_data(self.rng)

        start = time.time()

        # Impute potential nan values with the feature-
        data = np.concatenate((self.x_train, self.x_valid))
        targets = np.concatenate((self.y_train, self.y_valid))

        model = self._get_pipeline(n_estimators=fidelity['n_estimators'], **configuration)
        model.fit(X=data, y=targets)

        test_loss = 1 - self.accuracy_scorer(model, self.x_test, self.y_test)
        cost = time.time() - start

        return {'function_value': float(test_loss),
                'cost': cost,
                'info': {'fidelity': fidelity}}

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the XGBoost Model

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
            CS.UniformFloatHyperparameter('eta', lower=2**-10, upper=1., default_value=0.3, log=True),
            CS.UniformIntegerHyperparameter('max_depth', lower=1, upper=15, default_value=6, log=False),
            CS.UniformFloatHyperparameter('min_child_weight', lower=1., upper=2**7., default_value=1., log=True),
            CS.UniformFloatHyperparameter('colsample_bytree', lower=0.01, upper=1., default_value=1.),
            CS.UniformFloatHyperparameter('colsample_bylevel', lower=0.01, upper=1., default_value=1.),
            CS.UniformFloatHyperparameter('reg_lambda', lower=2**-10, upper=2**10, default_value=1, log=True),
            CS.UniformFloatHyperparameter('reg_alpha', lower=2**-10, upper=2**10, default_value=1, log=True),
            CS.UniformFloatHyperparameter('subsample_per_it', lower=0.1, upper=1, default_value=1, log=False)
        ])

        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the XGBoost Benchmark

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
            CS.UniformFloatHyperparameter("dataset_fraction", lower=0.0, upper=1.0, default_value=1.0, log=False),
            CS.UniformIntegerHyperparameter("n_estimators", lower=1, upper=256, default_value=256, log=False)
        ])

        return fidel_space

    def get_meta_information(self) -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'XGBoost',
                'references': ['@article{probst2019tunability,'
                               'title={Tunability: Importance of hyperparameters of machine learning algorithms.},'
                               'author={Probst, Philipp and Boulesteix, Anne-Laure and Bischl, Bernd},'
                               'journal={J. Mach. Learn. Res.},'
                               'volume={20},'
                               'number={53},'
                               'pages={1--32},'
                               'year={2019}'
                               '}'],
                'code': 'https://github.com/automl/HPOlib1.5/blob/development/hpolib/benchmarks/ml/'
                        'xgboost_benchmark_old.py',
                'shape of train data': self.x_train.shape,
                'shape of test data': self.x_test.shape,
                'shape of valid data': self.x_valid.shape,
                'initial random seed': self.rng,
                'task_id': self.task_id
                }

    def _get_pipeline(self, max_depth: int, eta: float, min_child_weight: int,
                      colsample_bytree: float, colsample_bylevel: float, reg_lambda: int, reg_alpha: int,
                      n_estimators: int, subsample_per_it: float) \
            -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """
        objective = 'binary:logistic' if self.num_class <= 2 else 'multi:softmax'

        clf = pipeline.Pipeline([
            ('preprocess_impute',
             ColumnTransformer([
                 ("categorical", "passthrough", self.categorical_data),
                 ("continuous", SimpleImputer(strategy="mean"), ~self.categorical_data)])),
            ('preprocess_one_hot',
             ColumnTransformer([
                 ("categorical", OneHotEncoder(categories=self.categories, sparse=False), self.categorical_data),
                 ("continuous", "passthrough", ~self.categorical_data)])),
            ('xgb',
             xgb.XGBClassifier(
                 max_depth=max_depth,
                 learning_rate=eta,
                 min_child_weight=min_child_weight,
                 colsample_bytree=colsample_bytree,
                 colsample_bylevel=colsample_bylevel,
                 reg_alpha=reg_alpha,
                 reg_lambda=reg_lambda,
                 n_estimators=n_estimators,
                 objective=objective,
                 n_jobs=self.n_threads,
                 random_state=self.rng.randint(1, 100000),
                 num_class=self.num_class,
                 subsample=subsample_per_it))
            ])
        return clf


class XGBoostExtendedBenchmark(XGBoostBenchmark):
    """
    Similar to XGBoostBenchmark but enables also the optimization of the used booster.
    """

    def __init__(self, task_id: Union[int, None] = None, n_threads: int = 1,
                 rng: Union[np.random.RandomState, int, None] = None):
        super(XGBoostExtendedBenchmark, self).__init__(task_id=task_id, n_threads=n_threads, rng=rng)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = XGBoostBenchmark.get_configuration_space(seed)
        hp_booster = CS.CategoricalHyperparameter('booster', choices=['gbtree', 'gblinear', 'dart'],
                                                  default_value='gbtree')
        cs.add_hyperparameter(hp_booster)

        # XGBoost with 'gblinear' can not use some
        # parameters. Exclude them from the configuration space by introducing a condition.
        hps = ['colsample_bylevel', 'colsample_bytree', 'max_depth',  'min_child_weight', 'subsample_per_it']

        # The NotEqualsCondition means: "Make parameter X active if hp_booster is not equal to gblinear."
        conditions = [CS.NotEqualsCondition(cs.get_hyperparameter(hp), hp_booster, 'gblinear') for hp in hps]
        cs.add_conditions(conditions)
        return cs

    # noinspection PyMethodOverriding
    # pylint: disable=arguments-differ
    def _get_pipeline(self, n_estimators: int, booster: str, reg_lambda: int, reg_alpha: int, eta: float,
                      min_child_weight: int = None, max_depth: int = None, colsample_bytree: float = None,
                      colsample_bylevel: float = None, subsample_per_it: float = None) \
            -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """
        objective = 'binary:logistic' if self.num_class <= 2 else 'multi:softmax'

        configuration = dict(booster=booster,
                             max_depth=max_depth,
                             learning_rate=eta,
                             min_child_weight=min_child_weight,
                             colsample_bytree=colsample_bytree,
                             colsample_bylevel=colsample_bylevel,
                             reg_alpha=reg_alpha,
                             reg_lambda=reg_lambda,
                             n_estimators=n_estimators,
                             objective=objective,
                             n_jobs=self.n_threads,
                             random_state=self.rng.randint(1, 100000),
                             num_class=self.num_class,
                             subsample=subsample_per_it)

        configuration = {k: v for k, v in configuration.items() if v is not None}

        clf = pipeline.Pipeline([
            ('preprocess_impute',
             ColumnTransformer([
                 ("categorical", "passthrough", self.categorical_data),
                 ("continuous", SimpleImputer(strategy="mean"), ~self.categorical_data)])),
            ('preprocess_one_hot',
             ColumnTransformer([
                 ("categorical", OneHotEncoder(categories=self.categories, sparse=False), self.categorical_data),
                 ("continuous", "passthrough", ~self.categorical_data)])),
            ('xgb',
             xgb.XGBClassifier(**configuration))
        ])
        return clf
