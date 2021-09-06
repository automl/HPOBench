"""

Changelog:
==========
0.0.3
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.2:
* Standardize the structure of the meta information

0.0.1:
* First implementation

"""

import logging
import time
from typing import Union, Tuple, Dict, List

import ConfigSpace as CS
import numpy as np
from scipy import sparse
from sklearn import pipeline
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.openml_data_manager import OpenMLHoldoutDataManager

__version__ = '0.0.3'

logger = logging.getLogger('SVMBenchmark')


class SupportVectorMachine(AbstractBenchmark):
    """
    Hyperparameter optimization task to optimize the regularization
    parameter C and the kernel parameter gamma of a support vector machine.
    Both hyperparameters are optimized on a log scale in [-10, 10].
    The X_test data set is only used for a final offline evaluation of
    a configuration. For that the validation and training data is
    concatenated to form the whole training data set.
    """

    def __init__(self, task_id: Union[int, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        task_id : int, None
        rng : np.random.RandomState, int, None
        """
        super(SupportVectorMachine, self).__init__(rng=rng)

        self.task_id = task_id
        self.cache_size = 200  # Cache for the SVC in MB
        self.accuracy_scorer = make_scorer(accuracy_score)

        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, variable_types = \
            self.get_data()
        self.categorical_data = np.array([var_type == 'categorical' for var_type in variable_types])

        # Sort data (Categorical + numerical) so that categorical and continous are not mixed.
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
        Trains a SVM model given a hyperparameter configuration and
        evaluates the model on the validation set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the SVM model
        fidelity: Dict, None
            Fidelity parameters for the SVM model, check get_fidelity_space(). Uses default (max) value if None.
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
                train_loss : training loss
                fidelity : used fidelities in this evaluation
        """
        start_time = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if shuffle:
            self.shuffle_data(self.rng)

        # Split of dataset subset
        if self.lower_bound_train_size > fidelity['dataset_fraction']:
            train_size = self.lower_bound_train_size
            logger.warning(f'The given data set fraction is lower than the lower bound (10 * number of classes.) '
                           f'Increase the fidelity from {fidelity["dataset_fraction"]:.8f} to '
                           f'{self.lower_bound_train_size:.8f}')
        else:
            train_size = fidelity['dataset_fraction']

        train_size = int(train_size * len(self.train_idx))
        train_idx = self.train_idx[:train_size]

        # Transform hyperparameters to linear scale
        hp_c = np.exp(float(configuration['C']))
        hp_gamma = np.exp(float(configuration['gamma']))

        # Train support vector machine
        model = self.get_pipeline(hp_c, hp_gamma)
        model.fit(self.x_train[train_idx], self.y_train[train_idx])

        # Compute validation error
        train_loss = 1 - self.accuracy_scorer(model, self.x_train[train_idx], self.y_train[train_idx])
        val_loss = 1 - self.accuracy_scorer(model, self.x_valid, self.y_valid)

        cost = time.time() - start_time

        return {'function_value': float(val_loss),
                "cost": cost,
                'info': {'train_loss': float(train_loss),
                         'fidelity': fidelity}}

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                shuffle: bool = False,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a SVM model with a given configuration on both the X_train
        and validation data set and evaluates the model on the X_test data set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the SVM Model
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
            function_value : X_test loss
            cost : time to X_train and evaluate the model
            info : Dict
                train_valid_loss: Loss on the train+valid data set
                fidelity : used fidelities in this evaluation
        """
        assert np.isclose(fidelity['dataset_fraction'], 1), \
            f'Data set fraction must be 1 but was {fidelity["dataset_fraction"]}'

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)

        if shuffle:
            self.shuffle_data(self.rng)

        start_time = time.time()

        # Concatenate training and validation dataset
        if isinstance(self.x_train, sparse.csr.csr_matrix) or isinstance(self.x_valid, sparse.csr.csr_matrix):
            data = sparse.vstack((self.x_train, self.x_valid))
        else:
            data = np.concatenate((self.x_train, self.x_valid))
        targets = np.concatenate((self.y_train, self.y_valid))

        # Transform hyperparameters to linear scale
        hp_c = np.exp(float(configuration['C']))
        hp_gamma = np.exp(float(configuration['gamma']))

        model = self.get_pipeline(hp_c, hp_gamma)
        model.fit(data, targets)

        # Compute validation error
        train_valid_loss = 1 - self.accuracy_scorer(model, data, targets)

        # Compute test error
        test_loss = 1 - self.accuracy_scorer(model, self.x_test, self.y_test)

        cost = time.time() - start_time

        return {'function_value': float(test_loss),
                "cost": cost,
                'info': {'train_valid_loss': float(train_valid_loss),
                         'fidelity': fidelity}}

    def get_pipeline(self, C: float, gamma: float) -> pipeline.Pipeline:
        """ Create the scikit-learn (training-)pipeline """

        model = pipeline.Pipeline([
            ('preprocess_impute',
             ColumnTransformer([
                 ("categorical", "passthrough", self.categorical_data),
                 ("continuous", SimpleImputer(strategy="mean"), ~self.categorical_data)])),
            ('preprocess_one_hot',
             ColumnTransformer([
                 ("categorical", OneHotEncoder(categories=self.categories, sparse=False), self.categorical_data),
                 ("continuous", MinMaxScaler(feature_range=(0, 1)), ~self.categorical_data)])),
            ('svm',
             svm.SVC(gamma=gamma, C=C, random_state=self.rng, cache_size=self.cache_size))
        ])
        return model

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the SVM Model

        For a detailed explanation of the hyperparameters:
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

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
            CS.UniformFloatHyperparameter('C', lower=-10., upper=10., default_value=0., log=False),
            CS.UniformFloatHyperparameter('gamma', lower=-10., upper=10., default_value=1., log=False),
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
        dataset_fraction: float - [0.1, 1]
            fraction of training data set to use

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
        ])
        return fidel_space

    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        return {'name': 'Support Vector Machine',
                'references': ["@InProceedings{pmlr-v54-klein17a",
                               "author = {Aaron Klein and Stefan Falkner and Simon Bartels and Philipp Hennig and "
                               "Frank Hutter}, "
                               "title = {{Fast Bayesian Optimization of Machine Learning Hyperparameters on "
                               "Large Datasets}}"
                               "pages = {528--536}, year = {2017},"
                               "editor = {Aarti Singh and Jerry Zhu},"
                               "volume = {54},"
                               "series = {Proceedings of Machine Learning Research},"
                               "address = {Fort Lauderdale, FL, USA},"
                               "month = {20--22 Apr},"
                               "publisher = {PMLR},"
                               "pdf = {http://proceedings.mlr.press/v54/klein17a/klein17a.pdf}, "
                               "url = {http://proceedings.mlr.press/v54/klein17a.html}, "
                               ],
                'code': 'https://github.com/automl/HPOlib1.5/blob/container/hpolib/benchmarks/ml/svm_benchmark.py',
                'shape of train data': self.x_train.shape,
                'shape of test data': self.x_test.shape,
                'shape of valid data': self.x_valid.shape,
                'initial random seed': self.rng,
                'task_id': self.task_id
                }
