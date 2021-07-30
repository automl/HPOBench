import abc
import time
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.dependencies.od.data_manager import OutlierDetectionDataManager
from hpobench.dependencies.od.utils.scaler import get_fitted_scaler


class ODTraditional(AbstractBenchmark):
    """
    Hyperparameter optimization task to optimize arbitrary traditional
    models for outlier detection.
    """

    def __init__(self,
                 dataset_name: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        dataset_name : str
        rng : np.random.RandomState, int, None
        """
        self.rng = rng_helper.get_rng(rng)

        # Load dataset manager
        self.dataset_name = dataset_name
        self.datamanager = OutlierDetectionDataManager(dataset_name, self.rng)

        super(ODTraditional, self).__init__(rng=self.rng)

    @abc.abstractmethod
    def get_name(self):
        """Returns the name of the model for the meta information."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model(self, configuration):
        """Returns the unfitted model given a configuration."""
        raise NotImplementedError()

    @abc.abstractmethod
    def calculate_scores(self, model, X):
        """Calculates the scores based on the model and X."""
        raise NotImplementedError()

    def calculate_aupr(self, model, X, y):
        """Calculates the AUPR based on the model, X and y."""
        scores = self.calculate_scores(model, X)

        precision, recall, thresholds = precision_recall_curve(y, scores)
        area = auc(recall, precision)

        return area

    @AbstractBenchmark.check_parameters
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a traditional model on a given hyperparameter configuration and
        evaluates the model on the validation set. 4-fold cross-validation is used.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the SVM model
        fidelity: Dict, None
            Fidelity parameters for the SVM model, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : 1 - mean of the validation AUPRs from each split.
            cost : time to train and evaluate the model
            info : Dict
                train_loss : 1 - mean of the train AUPRS from each split.
                fidelity : used fidelities in this evaluation
        """
        start_time = time.time()

        # Train support vector machine
        model = self.get_model(configuration)

        train_auprs = []
        val_auprs = []

        for split in range(4):
            (X_train, y_train), (X_val, y_val) = self.datamanager.dataset.get_train_val_data(split=split)

            # Normalize data
            scaler = get_fitted_scaler(X_train, configuration["scaler"])
            if scaler is not None:
                X_train = scaler(X_train)
                X_val = scaler(X_val)

            model.fit(X_train, y_train)

            # Compute train+validation error
            train_aupr = self.calculate_aupr(model, X_train, y_train)
            val_aupr = self.calculate_aupr(model, X_val, y_val)

            train_auprs.append(train_aupr)
            val_auprs.append(val_aupr)

        train_aupr = float(np.mean(np.array(train_auprs)))
        val_aupr = float(np.mean(np.array(val_auprs)))

        cost = time.time() - start_time

        return {
            'function_value': 1 - val_aupr,
            'cost': cost,
            'info': {
                'train_aupr': train_aupr,
                'val_aupr': val_aupr,
                'fidelity': fidelity
            }
        }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a traditional model with a given configuration on both the training
        and validation data set and evaluates the model on the X_test data set.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the SVM Model
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : 1 - AUPR (on test dataset)
            cost : time to X_train and evaluate the model
            info : Dict
                train_valid_loss: 1 - AUPR (on train+validation dataset)
                fidelity : used fidelities in this evaluation
        """

        start_time = time.time()

        X_train, y_train = self.datamanager.dataset.get_train_data()
        X_test, y_test = self.datamanager.dataset.get_test_data()

        # Normalize data
        scaler = get_fitted_scaler(X_train, configuration["scaler"])
        if scaler is not None:
            X_train = scaler(X_train)
            X_test = scaler(X_test)

        # Train support vector machine
        model = self.get_model(configuration)
        model.fit(X_train, y_train)

        # Compute validation error
        train_aupr = float(self.calculate_aupr(model, X_train, y_train))
        test_aupr = float(self.calculate_aupr(model, X_test, y_test))

        cost = time.time() - start_time

        return {
            'function_value': 1 - test_aupr,
            'cost': cost,
            'info': {
                'train_aupr': train_aupr,
                'test_aupr': test_aupr,
                'fidelity': fidelity
            }
        }

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates an empty ConfigSpace.ConfigurationSpace for traditional models
        as no fidelitie are used.

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
        return fidel_space

    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        X_train, y_train = self.datamanager.dataset.get_train_data()
        X_test, y_test = self.datamanager.dataset.get_test_data()

        return {
            'name': self.get_name(),
            'references': [
                '@misc{Rayana:2016 ,'
                'author = "Shebuti Rayana",'
                'year = "2016",'
                'title = “{ODDS} Library”,'
                'url = "http://odds.cs.stonybrook.edu",'
                'institution = "Stony Brook University, Department of Computer Sciences" }'

            ],
            'shape of train data': X_train.shape,
            'shape of test data': X_test.shape,
            'initial random seed': self.rng,
            'dataset_name': self.dataset_name,
            'contamination': self.datamanager.dataset.get_contamination_ratio()
        }
