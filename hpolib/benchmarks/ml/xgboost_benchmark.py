import time
from typing import Union, Tuple, Dict, List

import ConfigSpace as CS
import numpy as np
import xgboost as xgb
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import hpolib.util.rng_helper as rng_helper
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.openml_data_manager import OpenMLHoldoutDataManager

__version__ = '0.0.1'


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

        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, variable_types = \
            self.get_data()

        self.categorical_data = np.array([var_type == 'categorical' for var_type in variable_types])

        # Determine all possible values per categorical feature
        complete_data = np.concatenate([self.X_train, self.X_valid, self.X_test], axis=0)
        self.categories = [np.unique(complete_data[:, i])
                           for i in range(self.X_train.shape[1]) if self.categorical_data[i]]

        # Determine the number of categories in the labels.
        # In case of binary classification `self.num_class` has to be 1 for xgboost.
        self.num_class = len(np.unique(np.concatenate([self.y_train, self.y_test, self.y_valid])))
        self.num_class = 1 if self.num_class == 2 else self.num_class

        self.train_idx = self.rng.choice(a=np.arange(len(self.X_train)),
                                         size=len(self.X_train),
                                         replace=False)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """ Loads the data given a task or another source. """

        assert self.task_id is not None, NotImplementedError('No task-id given. Please either specify a task-id or '
                                                             'overwrite the get_data method.')

        dm = OpenMLHoldoutDataManager(openml_task_id=self.task_id, rng=self.rng)
        X_train, y_train, X_val, y_val, X_test, y_test = dm.load()

        return X_train, y_train, X_val, y_val, X_test, y_test, dm.variable_types

    def shuffle_data(self, rng=None):
        """ Reshuffle the training data. If 'rng' is None, the training idx are shuffled according to the
        class-random-state"""
        random_state = rng_helper.get_rng(rng, self.rng)
        random_state.shuffle(self.train_idx)

    @AbstractBenchmark._check_configuration
    def objective_function(self, config: Dict, n_estimators: int, subsample: float,
                           shuffle: bool = False, **kwargs) -> Dict:
        """
        Trains a XGBoost model given a hyperparameter configuration and
        evaluates the model on the validation set.

        To prevent overfitting on a single seed, it is possible to pass a
        parameter `rng` as 'int' or 'np.random.RandomState' to this function.
        If this parameter is not given, the default random state is used.

        Parameters
        ----------
        config : Dict
            Configuration for the XGBoost model
        n_estimators : int
            Number of trees to fit.
        subsample : float
            Subsample ratio of the training instance.
        shuffle : bool
            If `True`, shuffle the training idx. If no parameter `rng` is given, use the class random state.
            Defaults to `False`.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            train_loss : trainings loss
            subsample : fraction which was used to subsample the training data

        """
        assert 0 < subsample <= 1, ValueError(f'Parameter \'subsample\' must be in range (0, 1] but was {subsample}')

        rng = kwargs.get('rng', None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if shuffle:
            self.shuffle_data(self.rng)

        start = time.time()

        train_idx = self.train_idx[:int(len(self.train_idx) * subsample)]

        model = self._get_pipeline(n_estimators=n_estimators, **config)
        model.fit(X=self.X_train[train_idx], y=self.y_train[train_idx])

        train_loss = 1 - self.accuracy_scorer(model, self.X_train[train_idx], self.y_train[train_idx])
        val_loss = 1 - self.accuracy_scorer(model, self.X_valid, self.y_valid)
        cost = time.time() - start

        return {'function_value': val_loss,
                'cost': cost,
                'train_loss': train_loss,
                'subsample': subsample}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config: Dict, n_estimators: int, **kwargs) -> Dict:
        """
        Trains a XGBoost model with a given configuration on both the train
        and validation data set and evaluates the model on the test data set.

        To prevent overfitting on a single seed, it is possible to pass a
        parameter `rng` as 'int' or 'np.random.RandomState' to this function.
        If this parameter is not given, the default random state is used.

        Parameters
        ----------
        config : Dict
            Configuration for the XGBoost Model
        n_estimators : int
            Number of trees to fit.
        kwargs

        Returns
        -------
        Dict -
            function_value : test loss
            cost : time to train and evaluate the model
        """
        rng = kwargs.get('rng', None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        start = time.time()
        model = self._get_pipeline(n_estimators=n_estimators, **config)

        model.fit(X=np.concatenate((self.X_train, self.X_valid)),
                  y=np.concatenate((self.y_train, self.y_valid)))

        test_loss = 1 - self.accuracy_scorer(model, self.X_test, self.y_test)
        cost = time.time() - start

        return {'function_value': test_loss, 'cost': cost}

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
            CS.UniformFloatHyperparameter('eta', lower=1e-5, upper=1, default_value=0.3, log=True),
            CS.UniformFloatHyperparameter('min_child_weight', lower=0.05, upper=10., default_value=1., log=True),
            CS.UniformFloatHyperparameter('colsample_bytree', lower=0.05, upper=1., default_value=1.),
            CS.UniformFloatHyperparameter('colsample_bylevel', lower=0.05, upper=1., default_value=1.),
            CS.UniformFloatHyperparameter('reg_lambda', lower=1e-5, upper=2, default_value=1, log=True),
            CS.UniformFloatHyperparameter('reg_alpha', lower=1e-5, upper=2, default_value=1e-5, log=True)
        ])

        return cs

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'XGBoost',
                'references': ['None'],
                }

    def _get_pipeline(self, eta: float, min_child_weight: int, colsample_bytree: float, colsample_bylevel: float,
                      reg_lambda: int, reg_alpha: int, n_estimators: int) -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """
        objective = 'binary:logistic' if self.num_class <= 2 else 'multi:softmax'

        clf = pipeline.Pipeline(
            [('preprocess_impute',
              SimpleImputer(missing_values=np.nan, strategy='mean')),
             ('preprocess_one_hot',
              ColumnTransformer([
                 ("categorical", OneHotEncoder(categories=self.categories, sparse=False), self.categorical_data),
                 ("continuous", "passthrough", ~self.categorical_data)])),
             ('xgb', xgb.XGBClassifier(
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
                 num_class=self.num_class))
             ])
        return clf
