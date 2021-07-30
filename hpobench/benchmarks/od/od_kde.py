import logging
import time
from typing import Union, Tuple, Dict, List

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from sklearn.neighbors import KernelDensity

import hpobench.util.rng_helper as rng_helper
from hpobench.dependencies.od.traditional_benchmark import ODTraditional



__version__ = '0.0.1'

logger = logging.getLogger('ODKernelDensityEstimation')


class ODKernelDensityEstimation(ODTraditional):
    """
    Benchmark to train a Kernel Density Estimation (KDE) model for outlier detection. Overall,
    this benchmark can be used with one of 15 datasets (using a contamination ratio of 10%) provided by the ODDS Library (Rayana, 2016).
    Internally, a 4-fold cross-validation is used to prevent overfitting. Area under the precission-
    recall curve (AUPR) is used as metric.
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
        super(ODKernelDensityEstimation, self).__init__(
            dataset_name=dataset_name,
            rng=rng
        )

    def get_name(self):
        """Returns the name of the model for the meta information."""
        return "Kernel Density Estimation"
    
    def get_model(self, configuration):
        """Returns the unfitted model given a configuration."""
        hp_bandwidth = float(configuration['bandwidth'])

        return KernelDensity(kernel=configuration["kernel"], bandwidth=hp_bandwidth)

    def calculate_scores(self, model, X):
        """Calculates the scores based on the model and X."""
        return (-1.) * model.score_samples(X)

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

        bandwidth = CSH.UniformFloatHyperparameter('bandwidth', lower=pow(2, -5), upper=pow(2, 5), log=True)
        cs.add_hyperparameter(bandwidth)

        # Kernel
        kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
        choice = CSH.CategoricalHyperparameter(
            'kernel',
            kernels,
            default_value=kernels[0]
        )
        cs.add_hyperparameter(choice)

        # Scaler
        scalers = ["None", "MinMax", "Standard"]
        choice = CSH.CategoricalHyperparameter(
            'scaler',
            scalers,
            default_value=scalers[0]
        )
        cs.add_hyperparameter(choice)

        return cs
