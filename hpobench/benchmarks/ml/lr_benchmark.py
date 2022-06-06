"""
Changelog:
==========

0.0.1:
* First implementation of the LR Benchmarks.
0.0.2:
* Restructuring for consistency and to match ML Benchmark Template updates.
0.0.3:
* Adding Learning Curve support.
0.0.4:
* Extending to multi-objective query.
"""

import time
from typing import Union, Tuple, Dict, List

import ConfigSpace as CS
import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.linear_model import SGDClassifier

from hpobench.util.rng_helper import get_rng
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

__version__ = '0.0.4'


class LRBenchmark(MLBenchmark):
    """ Multi-multi-fidelity Logisitic Regression Benchmark
    """
    def __init__(
            self,
            task_id: int,
            valid_size: float = 0.33,
            rng: Union[np.random.RandomState, int, None] = None,
            data_path: Union[str, None] = None
    ):
        super(LRBenchmark, self).__init__(task_id, valid_size, rng, data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                "alpha", 1e-5, 1, log=True, default_value=1e-3
            ),
            CS.UniformFloatHyperparameter(
                "eta0", 1e-5, 1, log=True, default_value=1e-2
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - iterations + data subsample
            LRBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(
            iter_choice: str, subsample_choice: str
    ) -> Tuple[Hyperparameter, Hyperparameter]:
        """Fidelity space available --- specifies the fidelity dimensions
        """
        assert iter_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            fixed=CS.Constant('iter', value=1000),
            variable=CS.UniformIntegerHyperparameter(
                'iter', lower=10, upper=1000, default_value=1000, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1.0),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1.0, default_value=1.0, log=False
            )
        )
        iter = fidelity1[iter_choice]
        subsample = fidelity2[subsample_choice]
        return iter, subsample

    def init_model(
            self,
            config: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            rng: Union[int, np.random.RandomState, None] = None
    ):
        # initializing model
        rng = self.rng if rng is None else rng

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        # https://scikit-learn.org/stable/modules/sgd.html
        model = SGDClassifier(
            **config,
            loss="log",  # performs Logistic Regression
            max_iter=fidelity["iter"],
            learning_rate="adaptive",
            tol=None,
            random_state=rng,
        )
        return model

    def get_model_size(self, model: SGDClassifier = None) -> float:
        """ Returns the dimensionality as a proxy for the number of model parameters

        Logistic Regression models have a fixed number of parameters given a dataset. Model size is
        being approximated as the number of beta parameters required as the model support plus the
        intercept. This depends on the dataset and not on the trained model.

        Parameters
        ----------
        model : SGDClassifier
            Trained LR model. This parameter is required to maintain function signature.

        Returns
        -------
        float
        """
        ndims = self.train_X.shape[1]
        # accounting for the intercept
        ndims += 1
        return ndims

    def _train_objective(
            self,
            config: Dict,
            fidelity: Dict,
            shuffle: bool,
            rng: Union[np.random.RandomState, int, None] = None,
            evaluation: Union[str, None] = "valid",
            record_stats: bool = False,
            get_learning_curve: bool = False,
            lc_every_k: int = 1,
            **kwargs
    ):
        """Function that instantiates a 'config' on a 'fidelity' and trains it

        The ML model is instantiated and trained on the training split. Optionally, the model is
        evaluated on the training set. Optionally, the learning curves are collected.

        Parameters
        ----------
        config : CS.Configuration, Dict
            The hyperparameter configuration.
        fidelity : CS.Configuration, Dict
            The fidelity configuration.
        shuffle : bool (optional)
            If True, shuffles the training split before fitting the ML model.
        rng : np.random.RandomState, int (optional)
            The random seed passed to the ML model and if applicable, used for shuffling the data
            and subsampling the dataset fraction.
        evaluation : str (optional)
            If "valid", the ML model is trained on the training set alone.
            If "test", the ML model is trained on the training + validation sets.
        record_stats : bool (optional)
            If True, records the evaluation metrics of the trained ML model on the training set.
            This is set to False by default to reduce overall compute time.
        get_learning_curve : bool (optional)
            If True, records the learning curve using partial_fit or warm starting, if applicable.
            This is set to False by default to reduce overall compute time.
            Enabling True, implies that the for each iteration, the model will be evaluated on both
            the validation and test sets, optionally on the training set also.
        lc_every_k : int (optional)
            If True, records the learning curve after every k iterations.
        """
        if rng is not None:
            rng = get_rng(rng, self.rng)

        # initializing model
        model = self.init_model(config, fidelity, rng)

        # preparing data
        if evaluation == "valid":
            train_X = self.train_X
            train_y = self.train_y
        elif evaluation == "test":
            train_X = np.vstack((self.train_X, self.valid_X))
            train_y = pd.concat((self.train_y, self.valid_y))
        else:
            raise ValueError("{} not in ['valid', 'test']".format(evaluation))
        train_idx = np.arange(len(train_X)) if self.train_idx is None else self.train_idx

        # shuffling data
        if shuffle:
            train_idx = self.shuffle_data_idx(train_idx, rng)
            if isinstance(train_idx, np.ndarray):
                train_X = train_X[train_idx]
            else:
                train_X = train_X.iloc[train_idx]
            train_y = train_y.iloc[train_idx]

        # subsample here:
        # application of the other fidelity to the dataset that the model interfaces
        # carried over from previous HPOBench code that borrowed from FABOLAS' SVM
        lower_bound_lim = 1.0 / 512.0
        if self.lower_bound_train_size is None:
            self.lower_bound_train_size = (10 * self.n_classes) / self.train_X.shape[0]
            self.lower_bound_train_size = np.max((lower_bound_lim, self.lower_bound_train_size))
        subsample = np.max((fidelity['subsample'], self.lower_bound_train_size))
        train_idx = self.rng.choice(
            np.arange(len(train_X)), size=int(
                subsample * len(train_X)
            )
        )
        # fitting the model with subsampled data
        if get_learning_curve:
            # IMPORTANT to allow partial_fit
            model.warm_start = True
            lc_time = 0.0
            model_fit_time = 0.0
            learning_curves = dict(train=[], valid=[], test=[])
            lc_spacings = self._get_lc_spacing(model.max_iter, lc_every_k)
            iter_start = 0
            for i in range(len(lc_spacings)):
                iter_end = lc_spacings[i]
                start = time.time()
                # trains model for k steps
                for j in range(iter_end - iter_start):
                    model.partial_fit(
                        train_X[train_idx],
                        train_y.iloc[train_idx],
                        np.unique(train_y.iloc[train_idx])
                    )
                # adding all partial fit times
                model_fit_time += time.time() - start
                iter_start = iter_end
                lc_start = time.time()
                if record_stats:
                    train_pred = model.predict(train_X)
                    train_loss = 1 - self.scorers['acc'](
                        train_y, train_pred, **self.scorer_args['acc']
                    )
                    learning_curves['train'].append(train_loss)
                val_pred = model.predict(self.valid_X)
                val_loss = 1 - self.scorers['acc'](
                    self.valid_y, val_pred, **self.scorer_args['acc']
                )
                learning_curves['valid'].append(val_loss)
                test_pred = model.predict(self.test_X)
                test_loss = 1 - self.scorers['acc'](
                    self.test_y, test_pred, **self.scorer_args['acc']
                )
                learning_curves['test'].append(test_loss)
                # sums the time taken to evaluate and collect data for the learning curves
                lc_time += time.time() - lc_start
        else:
            # default training as per the base benchmark template
            learning_curves = None
            lc_time = None
            start = time.time()
            model.fit(train_X[train_idx], train_y.iloc[train_idx])
            model_fit_time = time.time() - start
        # model inference
        inference_time = 0.0
        # can optionally not record evaluation metrics on training set to save compute
        if record_stats:
            start = time.time()
            pred_train = model.predict(train_X)
            inference_time = time.time() - start
        # computing statistics on training data
        scores = dict()
        score_cost = dict()
        for k, v in self.scorers.items():
            scores[k] = 0.0
            score_cost[k] = 0.0
            _start = time.time()
            if record_stats:
                scores[k] = v(train_y, pred_train, **self.scorer_args[k])
            score_cost[k] = time.time() - _start + inference_time
        train_loss = 1 - scores["acc"]
        return model, model_fit_time, train_loss, scores, score_cost, learning_curves, lc_time


class LRBenchmarkBB(LRBenchmark):
    """ Black-box version of the LRBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            LRBenchmark._get_fidelity_choices(iter_choice='fixed', subsample_choice='fixed')
        )
        return fidelity_space


class LRBenchmarkMF(LRBenchmark):
    """ Multi-fidelity version of the LRBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - iterations
            LRBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space


class LRMOBenchmark(LRBenchmark):
    def __init__(
            self,
            task_id: int,
            valid_size: float = 0.33,
            rng: Union[np.random.RandomState, int, None] = None,
            data_path: Union[str, None] = None
    ):
        super(LRMOBenchmark, self).__init__(task_id, valid_size, rng, data_path)

    def get_objective_names(self):
        return ["loss", "inference_time"]

    def _get_multiple_objectives(self, result):
        single_obj = result['function_value']
        seeds = result['info'].keys()
        total_inference_time = sum([result['info']['val_costs']['acc']])
        avg_inference_time = total_inference_time / len(seeds)
        result['function_value'] = dict(
            loss=single_obj,
            inference_time=avg_inference_time
        )
        return result

    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            record_train: bool = False,
            get_learning_curve: bool = False,
            lc_every_k: int = 1,
            **kwargs
    ):
        result = super(LRMOBenchmark, self).objective_function(
            configuration=configuration,
            fidelity=fidelity,
            shuffle=shuffle,
            rng=rng,
            record_train=record_train,
            get_learning_curve=get_learning_curve,
            lc_every_k=lc_every_k,
            **kwargs
        )
        result = self._get_multiple_objectives(result)
        return result

    def objective_function_test(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            record_train: bool = False,
            get_learning_curve: bool = False,
            lc_every_k: int = 1,
            **kwargs
    ):
        result = super(LRMOBenchmark, self).objective_function_test(
            configuration=configuration,
            fidelity=fidelity,
            shuffle=shuffle,
            rng=rng,
            record_train=record_train,
            get_learning_curve=get_learning_curve,
            lc_every_k=lc_every_k,
            **kwargs
        )
        result = self._get_multiple_objectives(result)
        return result


class LRMOBenchmarkBB(LRMOBenchmark):
    """ Multi-fidelity version of the LRMOBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - iterations
            LRBenchmark._get_fidelity_choices(iter_choice='fixed', subsample_choice='fixed')
        )
        return fidelity_space


class LRMOBenchmarkMF(LRMOBenchmark):
    """ Multi-fidelity version of the LRBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - iterations
            LRBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space


__all__ = [
    'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
    'LRMOBenchmark', 'LRMOBenchmarkBB', 'LRMOBenchmarkMF'
]