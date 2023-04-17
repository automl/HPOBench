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
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark, AbstractSingleObjectiveBenchmark
from typing import Union, Dict, List, Text, Tuple

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


class _MLBenchmarkBase():
    _issue_tasks = [3917, 3945]

    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, Path, None] = None,
            global_seed: int = 1
    ):
        super(_MLBenchmarkBase, self).__init__(rng=rng)

        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)

        self.global_seed = global_seed  # used for fixed training-validation splits

        self.task_id = task_id
        self.valid_size = valid_size
        self.scorers = dict()
        for k, v in metrics.items():
            self.scorers[k] = make_scorer(v, **metrics_kwargs[k])

        if data_path is None:
            from hpobench import config_file
            data_path = config_file.data_dir / "OpenML"

        self.data_path = Path(data_path)

        dm = OpenMLDataManager(task_id, valid_size, data_path, global_seed)
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

        If fidelity_choice is 0
            Fidelity space is the maximal fidelity, akin to a black-box function
        If fidelity_choice is 1
            Fidelity space is a single fidelity, in this case the number of trees (n_estimators)
        If fidelity_choice is 2
            Fidelity space is a single fidelity, in this case the fraction of dataset (subsample)
        If fidelity_choice is >2
            Fidelity space is multi-multi fidelity, all possible fidelities
        """
        raise NotImplementedError()

    # pylint: disable=arguments-differ
    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Support Vector Machine',
            'shape of train data': self.train_X.shape,
            'shape of test data': self.test_X.shape,
            'shape of valid data': self.valid_X.shape,
            'initial random seed': self.seed,
            'task_id': self.task_id
        }

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
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

    def shuffle_data_idx(self, train_idx: Iterable = None, rng: Union[np.random.RandomState, None] = None) -> Iterable:
        rng = self.rng if rng is None else rng
        train_idx = self.train_idx if train_idx is None else train_idx
        rng.shuffle(train_idx)
        return train_idx

    def _train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid"):

        if rng is not None:
            rng = get_rng(rng, self.rng)

        # initializing model
        model = self.init_model(config, fidelity, rng)

        # preparing data
        if eval == "valid":
            train_X = self.train_X
            train_y = self.train_y
            train_idx = self.train_idx
        else:
            train_X = np.vstack((self.train_X, self.valid_X))
            train_y = pd.concat((self.train_y, self.valid_y))
            train_idx = np.arange(len(train_X))

        # shuffling data
        if shuffle:
            train_idx = self.shuffle_data_idx(train_idx, rng)
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
        # computing statistics on training data
        scores = dict()
        score_cost = dict()
        for k, v in self.scorers.items():
            scores[k] = 0.0
            score_cost[k] = 0.0
            if evaluation == "test":
                _start = time.time()
                scores[k] = v(model, train_X, train_y)
                score_cost[k] = time.time() - _start
        train_loss = 1 - scores["acc"]
        return model, model_fit_time, train_loss, scores, score_cost
    
    @AbstractMultiObjectiveBenchmark.check_parameters
    def _mo_objective_function(self,
                            configuration: Union[CS.Configuration, Dict],
                            fidelity: Union[CS.Configuration, Dict, None] = None,
                            shuffle: bool = False,
                            rng: Union[np.random.RandomState, int, None] = None,
                            **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
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
            function_value :
                ToDo: add return values 
            cost : runtime
            info : Dict
                fidelity : used fidelities in this evaluation
        
        """
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="val"
        )
        val_scores = dict()
        val_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            val_scores[k] = v(model, self.valid_X, self.valid_y)
            val_score_cost[k] = time.time() - _start
        val_loss = 1 - val_scores["acc"]
        val_sensitivity = (val_scores['f1'] * val_scores['precision']) /  (2 * val_scores['precision'] - val_scores['f1'] )
        val_scores['fnr'] = 1 - val_sensitivity

        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            test_scores[k] = v(model, self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        test_loss = 1 - test_scores["acc"]

        test_sensitivity = (test_scores['f1'] * test_scores['precision']) /  (2 * test_scores['precision'] - test_scores['f1'] )
        test_scores['fnr'] = 1 - test_sensitivity

        info = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
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

        result_dict = {
            'function_value':  {
                # The original benchmark returned the accuracy with range [0, 100].
                # We cast it to a minimization problem with range [0-1] to have a more standardized return value.
                'val_loss':  val_loss,
                'fnr': val_scores['fnr'],

            },
            'cost': model_fit_time + info['val_costs']['acc'],
            'info': info
        }

        return result_dict

    @AbstractMultiObjectiveBenchmark.check_parameters
    def _mo_objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="test"
        )
        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            test_scores[k] = v(model, self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        test_loss = 1 - test_scores["acc"]

        test_sensitivity = (test_scores['f1'] * test_scores['precision']) /  (2 * test_scores['precision'] - test_scores['f1'] )
        test_scores['fnr'] = 1 - test_sensitivity

        info = {
            'train_loss': train_loss,
            'val_loss': None,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': dict(),
            'val_costs': dict(),
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return {
            'function_value':  {
                # The original benchmark returned the accuracy with range [0, 100].
                # We cast it to a minimization problem with range [0-1] to have a more standardized return value.
                'test_loss':  float(info['test_loss']),
                # 'f1': test_scores['f1'],
                # 'precision': test_scores['precision'],
                # 'balanced_accuracy': test_scores['bal_acc'],
                'fnr': test_scores['fnr'],
            },
            'cost': float(model_fit_time + info['test_costs']['acc']),
            'info': info
        }

    
class MLBenchmark(_MLBenchmarkBase, AbstractSingleObjectiveBenchmark):
    

    @AbstractSingleObjectiveBenchmark.check_parameters
    def objective_function(self,
                            configuration: Union[CS.Configuration, Dict],
                            fidelity: Union[CS.Configuration, Dict, None] = None,
                            shuffle: bool = False,
                            rng: Union[np.random.RandomState, int, None] = None,
                            **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
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
            function_value :
                ToDo: add return values 
            cost : runtime
            info : Dict
                fidelity : used fidelities in this evaluation
        
        """
        results = self._mo_objective_function(
            configuration=configuration, fidelity=fidelity, rng=rng,  **kwargs
        )
        results['function_value'] = results['function_value']['val_loss']
        return results

    @AbstractSingleObjectiveBenchmark.check_parameters
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """

        results = self._mo_objective_function_test(
            configuration=configuration, fidelity=fidelity, rng=rng,  **kwargs
        )
        results['function_value'] = results['function_value']['test_loss']
        return results

class MO_MLBenchmark(_MLBenchmarkBase, AbstractMultiObjectiveBenchmark):
    
    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self,
                            configuration: Union[CS.Configuration, Dict],
                            fidelity: Union[CS.Configuration, Dict, None] = None,
                            shuffle: bool = False,
                            rng: Union[np.random.RandomState, int, None] = None,
                            **kwargs) -> Dict:
        return self._mo_objective_function(configuration=configuration, fidelity=fidelity,shuffle=shuffle, rng=rng,
                                           **kwargs)

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        return self._mo_objective_function_test(configuration=configuration, fidelity=fidelity, shuffle=shuffle, rng=rng,
                                           **kwargs)
    @staticmethod
    def get_objective_names() -> List[str]:
        return ['val_loss', 'fnr']

