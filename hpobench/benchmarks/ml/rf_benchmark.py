import time
import openml
import numpy as np
import pandas as pd
import ConfigSpace as CS
from typing import Union, Dict

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class RandomForestBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1
    ):
        super(RandomForestBenchmark, self).__init__(task_id, seed, valid_size, fidelity_choice)
        pass

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=15, default_value=2, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'min_samples_split', lower=2, upper=128, default_value=2, log=True
            ),
            CS.UniformFloatHyperparameter(
                'max_features', lower=0.1, upper=0.9, default_value=0.5, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'min_samples_leaf', lower=1, upper=64, default_value=1, log=True
            ),
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed=None, fidelity_choice=1):
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
        z_cs = CS.ConfigurationSpace(seed=seed)
        subsample_lower_bound = np.max((0.1, (0.1 or self.lower_bound_train_size)))
        if fidelity_choice == 0:
            # only subsample as fidelity
            ntrees = CS.Constant('n_estimators', value=100)
            subsample = CS.Constant('subsample', value=1)
        elif fidelity_choice == 1:
            # only n_estimators as fidelity
            ntrees = CS.UniformIntegerHyperparameter(
                'n_estimators', lower=2, upper=100, default_value=10, log=False
            )
            subsample = CS.Constant('subsample', value=1)
        elif fidelity_choice == 2:
            # only subsample as fidelity
            ntrees = CS.Constant('n_estimators', value=100)
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=subsample_lower_bound, upper=1, default_value=1, log=False
            )
        else:
            # both n_estimators and subsample as fidelities
            ntrees = CS.UniformIntegerHyperparameter(
                'n_estimators', lower=2, upper=100, default_value=10, log=False
            )
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=subsample_lower_bound, upper=1, default_value=1, log=False
            )
        z_cs.add_hyperparameters([ntrees, subsample])
        return z_cs

    def init_model(self, config, fidelity=None, rng=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        model = RandomForestClassifier(
            **config.get_dictionary(),
            n_estimators=fidelity['n_estimators'],  # a fidelity being used during initialization
            bootstrap=True,
            random_state=rng
        )
        return model
