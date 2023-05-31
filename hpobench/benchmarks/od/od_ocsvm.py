import logging
from typing import Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from sklearn.svm import OneClassSVM

from hpobench.dependencies.od.traditional_benchmark import ODTraditional

__version__ = '0.0.1'

logger = logging.getLogger('ODOneClassSupportVectorMachine')


class ODOneClassSupportVectorMachine(ODTraditional):
    """
    Benchmark to train a One-Class Support Vector Machine (OC-SVM) model for outlier detection. Overall,
    this benchmark can be used with one of 15 datasets (using a contamination ratio of 10%) provided by the
    ODDS Library (Rayana, 2016). Internally, a 4-fold cross-validation is used to prevent overfitting.
    Area under the precission-recall curve (AUPR) is used as metric.
    """

    def __init__(self,
                 dataset_name: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        dataset_name : str
            Must be one of [
                "annthyroid", "arrhythmia", "breastw", "cardio", "ionosphere",
                "mammography", "musk", "optdigits", "pendigits", "pima",
                "satellite", "satimage-2", "thyroid", "vowels", "wbc"]
        rng : np.random.RandomState, int, None
        """
        super(ODOneClassSupportVectorMachine, self).__init__(
            dataset_name=dataset_name,
            rng=rng
        )

    def get_name(self):
        """Returns the name of the model for the meta information."""
        return "One Class Support Vector Machine"

    def get_model(self, configuration):
        """Returns the unfitted model given a configuration."""
        hp_gamma = float(configuration['gamma'])
        hp_nu = float(configuration['nu'])

        return OneClassSVM(kernel="rbf", gamma=hp_gamma, nu=hp_nu)

    def calculate_scores(self, model, X):
        """Calculates the scores based on the model and X."""
        return (-1) * model.decision_function(X)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the OCSVM Model.

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
            CS.UniformFloatHyperparameter('gamma', lower=pow(2, -20), upper=pow(2, -2), log=True),
            CS.UniformFloatHyperparameter('nu', lower=0.0, upper=1.0, default_value=0.5),
        ])

        # Scaler
        scalers = ["None", "MinMax", "Standard"]
        choice = CSH.CategoricalHyperparameter(
            'scaler',
            scalers,
            default_value=scalers[0]
        )
        cs.add_hyperparameter(choice)

        return cs
