import time
from pathlib import Path
from typing import Union, Dict, Iterable

import ConfigSpace as CS
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, \
    precision_score, f1_score

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.dependencies.ml.data_manager import OpenMLDataManager
from hpobench.util.rng_helper import get_rng

metrics = dict(
    acc=accuracy_score,
    bal_acc=balanced_accuracy_score,
    f1=f1_score,
    precision=precision_score,
)

metrics_kwargs = dict(
    acc=dict(),
    bal_acc=dict(),
    f1=dict(average="macro", zero_division=0),
    precision=dict(average="macro", zero_division=0),
)


class MLBenchmark(AbstractBenchmark):
    _issue_tasks = [3917, 3945]

    def __init__(
            self,
            task_id: int,
            valid_size: float = 0.33,
            rng: Union[np.random.RandomState, int, None] = None,
            data_path: Union[str, Path, None] = None,
            global_seed: int = 1
    ):
        """ Base template for the ML multi-fidelity benchmarks.

        Parameters
        ----------
        task_id : int
            A valid OpenML Task ID.
        valid_size : float
            The fraction of training set to be used as validation split.
        rng : np.random.RandomState, int (optional)
            The random seed that will be passed to the ML model if not explicitly passed.
        data_path : str, Path (optional)
            The path from where the training-validation-testing splits may be loaded.
        global_seed : int
            The fixed global seed that is used for creating validation splits if not available.
        """
        super(MLBenchmark, self).__init__(rng=rng)

        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)
        self.rng = get_rng(self.seed)

        self.global_seed = global_seed  # used for fixed training-validation splits

        self.task_id = task_id
        self.valid_size = valid_size
        self.scorers = metrics
        self.scorer_args = metrics_kwargs

        if data_path is None:
            from hpobench import config_file
            data_path = config_file.data_dir / "OpenML"

        self.data_path = Path(data_path)

        dm = OpenMLDataManager(self.task_id, self.valid_size, self.data_path, self.global_seed)
        dm.load()

        # Data variables
        self.train_X = dm.train_X
        self.valid_X = dm.valid_X
        self.test_X = dm.test_X
        self.train_y = dm.train_y
        self.valid_y = dm.valid_y
        self.test_y = dm.test_y
        self.train_idx = dm.train_idx
        self.test_idx = dm.test_idx
        self.task = dm.task
        self.dataset = dm.dataset
        self.preprocessor = dm.preprocessor
        self.lower_bound_train_size = dm.lower_bound_train_size
        self.n_classes = dm.n_classes

        # Observation and fidelity spaces
        self.fidelity_space = self.get_fidelity_space(self.seed)
        self.configuration_space = self.get_configuration_space(self.seed)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Fidelity space available --- specifies the fidelity dimensions
        """
        raise NotImplementedError()

    def get_meta_information(self):
        """ Returns the meta information for the benchmark
        """
        return {
            'name': 'Support Vector Machine',
            'shape of train data': self.train_X.shape,
            'shape of test data': self.test_X.shape,
            'shape of valid data': self.valid_X.shape,
            'initial random seed': self.rng,
            'task_id': self.task_id
        }

    def get_model_size(self, model):
        """ Returns a custom model size specific to the ML model, if applicable
        """
        raise NotImplementedError

    def init_model(
            self,
            config: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            rng: Union[int, np.random.RandomState, None] = None
    ):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        raise NotImplementedError()

    def get_config(self, size: Union[int, None] = None):
        """Samples configuration(s) from the (hyper) parameter space
        """
        if size is None:  # return only one config
            return self.configuration_space.sample_configuration()
        return [self.configuration_space.sample_configuration() for i in range(size)]

    def get_fidelity(self, size: Union[int, None] = None):
        """Samples candidate fidelities from the fidelity space
        """
        if size is None:  # return only one config
            return self.fidelity_space.sample_configuration()
        return [self.fidelity_space.sample_configuration() for i in range(size)]

    def shuffle_data_idx(
            self, train_idx: Iterable = None, rng: Union[np.random.RandomState, None] = None
    ) -> Iterable:
        rng = self.rng if rng is None else rng
        train_idx = self.train_idx if train_idx is None else train_idx
        rng.shuffle(train_idx)
        return train_idx

    def _train_objective(
            self,
            config: Dict,
            fidelity: Dict,
            shuffle: bool,
            rng: Union[np.random.RandomState, int, None] = None,
            evaluation: Union[str, None] = "valid",
            record_stats: bool = False
    ):
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
        if self.lower_bound_train_size is None:
            self.lower_bound_train_size = (10 * self.n_classes) / self.train_X.shape[0]
            self.lower_bound_train_size = np.max((1 / 512, self.lower_bound_train_size))
        subsample = np.max((fidelity['subsample'], self.lower_bound_train_size))
        train_idx = self.rng.choice(
            np.arange(len(train_X)), size=int(
                subsample * len(train_X)
            )
        )
        # fitting the model with subsampled data
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
        return model, model_fit_time, train_loss, scores, score_cost

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            record_train: bool = False,
            **kwargs
    ) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set

        The ML model is trained on the training split, and evaluated on the valid and test splits.

        Parameters
        ----------
        configuration : CS.Configuration, Dict
            The hyperparameter configuration.
        fidelity : CS.Configuration, Dict
            The fidelity configuration.
        shuffle : bool (optional)
            If True, shuffles the training split before fitting the ML model.
        rng : np.random.RandomState, int (optional)
            The random seed passed to the ML model and if applicable, used for shuffling the data
            and subsampling the dataset fraction.
        record_train : bool (optional)
            If True, records the evaluation metrics of the trained ML model on the training set.
            This is set to False by default to reduce overall compute time.
        """
        # obtaining model and training statistics
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="valid", record_stats=record_train
        )
        model_size = self.get_model_size(model)

        # model inference on validation set
        start = time.time()
        pred_val = model.predict(self.valid_X)
        val_inference_time = time.time() - start
        val_scores = dict()
        val_score_cost = dict()
        for k, v in self.scorers.items():
            val_scores[k] = 0.0
            val_score_cost[k] = 0.0
            _start = time.time()
            val_scores[k] = v(self.valid_y, pred_val, **self.scorer_args[k])
            val_score_cost[k] = time.time() - _start + val_inference_time
        val_loss = 1 - val_scores["acc"]

        # model inference on test set
        start = time.time()
        pred_test = model.predict(self.test_X)
        test_inference_time = time.time() - start
        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            test_scores[k] = 0.0
            test_score_cost[k] = 0.0
            _start = time.time()
            test_scores[k] = v(self.test_y, pred_test, **self.scorer_args[k])
            test_score_cost[k] = time.time() - _start + test_inference_time
        test_loss = 1 - test_scores["acc"]

        fidelity = fidelity.get_dictionary() if isinstance(fidelity, CS.Configuration) else fidelity
        configuration = configuration.get_dictionary() \
            if isinstance(configuration, CS.Configuration) else configuration

        info = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'model_size': model_size,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': val_scores,
            'val_costs': val_score_cost,
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value': float(info['val_loss']),
            'cost': float(model_fit_time + info['val_costs']['acc']),
            'info': info
        }

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function_test(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            record_train: bool = False,
            **kwargs
    ) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set

        The ML model is trained on the training+valid split, and evaluated on the test split.

        Parameters
        ----------
        configuration : CS.Configuration, Dict
            The hyperparameter configuration.
        fidelity : CS.Configuration, Dict
            The fidelity configuration.
        shuffle : bool (optional)
            If True, shuffles the training split before fitting the ML model.
        rng : np.random.RandomState, int (optional)
            The random seed passed to the ML model and if applicable, used for shuffling the data
            and subsampling the dataset fraction.
        record_train : bool (optional)
            If True, records the evaluation metrics of the trained ML model on the training set.
            This is set to False by default to reduce overall compute time.
        """
        # obtaining model and training statistics
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="test", record_stats=record_train
        )
        model_size = self.get_model_size(model)

        # model inference on test set
        start = time.time()
        pred_test = model.predict(self.test_X)
        test_inference_time = time.time() - start
        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            test_scores[k] = 0.0
            test_score_cost[k] = 0.0
            _start = time.time()
            test_scores[k] = v(self.test_y, pred_test, **self.scorer_args[k])
            test_score_cost[k] = time.time() - _start + test_inference_time
        test_loss = 1 - test_scores["acc"]

        fidelity = fidelity.get_dictionary() if isinstance(fidelity, CS.Configuration) else fidelity
        configuration = configuration.get_dictionary() \
            if isinstance(configuration, CS.Configuration) else configuration

        info = {
            'train_loss': train_loss,
            'val_loss': None,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'model_size': model_size,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': None,
            'val_costs': None,
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value': float(info['test_loss']),
            'cost': float(model_fit_time + info['test_costs']['acc']),
            'info': info
        }


if __name__ == "__main__":
    from hpobench.benchmarks.ml import XGBoostBenchmarkMF
    benchmark = XGBoostBenchmarkMF(task_id=10101)
    config = benchmark.configuration_space.sample_configuration()
    print(config)
    fidelity = benchmark.fidelity_space.sample_configuration()
    print(fidelity)
    res = benchmark.objective_function(config, fidelity, shuffle=True, record_train=True, rng=123)
    print(res)
