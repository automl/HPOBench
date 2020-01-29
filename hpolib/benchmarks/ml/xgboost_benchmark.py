import time
from typing import Union, Tuple

import ConfigSpace as CS
import numpy as np
import xgboost as xgb
from sklearn import preprocessing, pipeline
from sklearn.metrics import accuracy_score

import hpolib.util.rng_helper as rng_helper
from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util.data_manager import MNISTData
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

        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.get_data()

        self.train_idx = self.rng.choice(a=np.arange(len(self.X_train)), size=len(self.X_train), replace=False)
        self.valid_idx = self.rng.choice(a=np.arange(len(self.X_valid)), size=len(self.X_valid), replace=False)

    def get_data(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        """ Loads the data given a task or another source. """

        assert self.task_id is not None, NotImplementedError('No taskid given. Please either specify a'
                                                             'taskid or overwrite the get_data method.')

        dm = OpenMLHoldoutDataManager(openml_task_id=self.task_id, rng=self.rng)
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = dm.load()
        except ValueError as e:
            raise e  # Currently, only holdout-data-sets are supported

        return X_train, y_train, X_val, y_val, X_test, y_test

    @AbstractBenchmark._check_configuration
    # @AbstractBenchmark._configuration_as_array
    def objective_function(self, config: dict, n_estimators: int, subsample: float, **kwargs) -> dict:
        """
        Trains a XGBoost model given a hyperparameter configuration and evaluates the model on the validation set.

        To prevent overfitting on a single seed, it is possible to pass a parameter `rng`
        as 'int' or 'np.random.RandomState' to this function. If this parameter is not given, the default random state
        is used.

        Parameters
        ----------
        config : dict
            Configuration for the XGBoost model
        n_estimators : int
            Number of trees to fit.
        subsample : float
            Subsample ratio of the training instance.
        kwargs

        Returns
        -------
        dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            train_loss : trainings loss
            subsample : fraction which was used to subsample the training data

        """
        assert 0 < subsample <= 1, ValueError(f'Parameter \'subsample\' must be in range (0, 1] but was {subsample}')

        rng = kwargs.get('rng', None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        start = time.time()

        model = self._get_pipeline(n_estimators=n_estimators, **config)

        train_idx = self.train_idx[:int(len(self.train_idx) * subsample)]

        model.fit(X=self.X_train[train_idx], y=self.y_train[train_idx])

        train_loss = XGBoostBenchmark._eval_xgb(model=model, X=self.X_train[train_idx], y=self.y_train[train_idx])
        val_loss = XGBoostBenchmark._eval_xgb(model=model, X=self.X_valid, y=self.y_valid)
        cost = time.time() - start

        return {'function_value': val_loss, 'cost': cost, 'train_loss': train_loss, 'subsample': subsample}

    @AbstractBenchmark._check_configuration
    # @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, config: dict, n_estimators: int, subsample: float, **kwargs) -> dict:
        """
        Trains a XGBoost model with a given configuration on both the train and validation data set and
        evaluates the model on the test data set.

        To prevent overfitting on a single seed, it is possible to pass a parameter `rng`
        as 'int' or 'np.random.RandomState' to this function. If this parameter is not given, the default random state
        is used.

        Parameters
        ----------
        config : dict
            Configuration for the XGBoost Model
        n_estimators : int
            Number of trees to fit.
        subsample : float
            Subsample ratio of the training instance.
        kwargs

        Returns
        -------
        dict -
            function_value : test loss
            cost : time to train and evaluate the model
            subsample : fraction which was used to subsample the training data
        """
        assert 0 < subsample <= 1, ValueError(f'Parameter \'subsample\' must be in range (0, 1] but was {subsample}')

        rng = kwargs.get('rng', None)
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        start = time.time()
        model = self._get_pipeline(n_estimators=n_estimators, **config)

        train_idx = self.train_idx[:int(len(self.train_idx) * subsample)]
        valid_idx = self.valid_idx[:int(len(self.valid_idx) * subsample)]

        model.fit(X=np.concatenate((self.X_train[train_idx], self.X_valid[valid_idx])),
                  y=np.concatenate((self.y_train[train_idx], self.y_valid[valid_idx])))

        test_loss = XGBoostBenchmark._eval_xgb(model=model, X=self.X_test, y=self.y_test)
        cost = time.time() - start

        return {'function_value': test_loss, 'cost': cost, 'subsample': subsample}

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for the XGBoost Model

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
            CS.UniformFloatHyperparameter('eta', lower=1e-3, upper=1, log=True),
            CS.UniformFloatHyperparameter('min_child_weight', lower=1e-10, upper=7., log=True),
            CS.UniformFloatHyperparameter('colsample_bytree', lower=0., upper=1.),
            CS.UniformFloatHyperparameter('colsample_bylevel', lower=0., upper=1.),
            CS.UniformFloatHyperparameter('reg_lambda', lower=2e-10, upper=2e10, log=True),
            CS.UniformFloatHyperparameter('reg_alpha', lower=2e-10, upper=2e10, log=True)
        ])

        return cs

    @staticmethod
    def get_meta_information() -> dict:
        """ Returns the meta information for the benchmark """
        from ConfigSpace.read_and_write.json import write as cs_to_json
        return {'name': 'XGBoost',
                'configuration space': cs_to_json(XGBoostBenchmark.get_configuration_space()),
                'references': ['None']
                }

    def _get_pipeline(self, eta: float, min_child_weight: int, colsample_bytree: float, colsample_bylevel: float,
                      reg_lambda: int, reg_alpha: int, n_estimators: int) -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """
        clf = pipeline.Pipeline([  # No normalizing as it should not make a difference
                                   # ('preproc', preprocessing.StandardScaler()),
                                 ('xgb', xgb.XGBClassifier(learning_rate=eta,
                                                           min_child_weight=min_child_weight,
                                                           colsample_bytree=colsample_bytree,
                                                           colsample_bylevel=colsample_bylevel,
                                                           reg_alpha=reg_alpha,
                                                           reg_lambda=reg_lambda,
                                                           n_estimators=n_estimators,
                                                           objective='binary:logistic',
                                                           n_jobs=self.n_threads,
                                                           random_state=self.rng.randint(1, 100000)))
                                ])
        return clf

    @staticmethod
    def _eval_xgb(X: np.array, y: np.array, model: Union[xgb.XGBModel, pipeline.Pipeline]) -> float:
        """ Helper-function for evaluating the XGBoost model. """
        y_pred = model.predict(X)
        acc = accuracy_score(y_pred=y_pred, y_true=y)
        return 1 - acc


class XGBoostOnHiggs(XGBoostBenchmark):

    def __init__(self, n_threads: int = 1, rng: Union[int, None] = None):
        super().__init__(task_id=75101, n_threads=n_threads, rng=rng)

    def _get_pipeline(self, eta: float, colsample_bytree: float, colsample_bylevel: float,
                      n_estimators: int, reg_lambda: int = 1, reg_alpha: int = 0,
                      min_child_weight: int = 1) -> pipeline.Pipeline:

        clf = pipeline.Pipeline([('preproc1', preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
                                 # No normalizing as it should not make a difference
                                 # ('preproc2', preprocessing.StandardScaler()),
                                 ('xgb', xgb.XGBClassifier(learning_rate=eta,
                                                           n_estimators=n_estimators,
                                                           objective='binary:logistic',
                                                           n_jobs=self.n_threads,
                                                           colsample_bytree=colsample_bytree,
                                                           colsample_bylevel=colsample_bylevel,
                                                           random_state=self.rng.randint(1, 100000)))
                                 ])
        return clf


class XGBoostOnMnist(XGBoostBenchmark):
    def __init__(self, n_threads: int = 1, rng: Union[int, np.random.RandomState, None] = None):
        super().__init__(task_id=None, n_threads=n_threads, rng=rng)

    def get_data(self):
        dm = MNISTData()
        return dm.load()


"""
class XGBoostOnMNIST2(XGBoostBenchmark):
    # This will currently fail, because the task consists of more than a
    # single fold
    def __init__(self, n_threads=1, rng=None):
        super().__init__(task_id=3573, n_threads=n_threads, rng=rng)
"""
