import time
import openml
import numpy as np
import pandas as pd
import ConfigSpace as CS
from typing import Union, Dict

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class XGBoostBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1
    ):
        super(XGBoostBenchmark, self).__init__(task_id, seed, valid_size, fidelity_choice)
        pass

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                'eta', lower=2**-10, upper=1., default_value=0.3, log=True
            ),  # learning rate
            CS.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=15, default_value=6, log=False
            ),
            CS.UniformFloatHyperparameter(
                'min_child_weight', lower=1., upper=2**7., default_value=1., log=True
            ),
            CS.UniformFloatHyperparameter(
                'colsample_bytree', lower=0.01, upper=1., default_value=1.
            ),
            # CS.UniformFloatHyperparameter(
            #     'colsample_bylevel', lower=0.01, upper=1., default_value=1.
            # ),
            CS.UniformFloatHyperparameter(
                'reg_lambda', lower=2**-10, upper=2**10, default_value=1, log=True
            ),
            # CS.UniformFloatHyperparameter(
            #     'reg_alpha', lower=2**-10, upper=2**10, default_value=1, log=True
            # ),
            # CS.UniformFloatHyperparameter(
            #     'subsample_per_it', lower=0.1, upper=1, default_value=1, log=False
            # )
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
        rng = rng if (rng is None and isinstance(rng, int)) else self.seed
        extra_args = dict(
            n_estimators=fidelity['n_estimators'],
            objective="binary:logistic",
            random_state=rng,
            subsample=1
        )
        if self.n_classes > 2:
            extra_args["objective"] = "multi:softmax"
            extra_args.update({"num_class": self.n_classes})
        model = xgb.XGBClassifier(
            **config.get_dictionary(),
            **extra_args
        )
        return model
