import time
import openml
import numpy as np
import pandas as pd
import ConfigSpace as CS
from typing import Union, Dict

from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, make_scorer

from hpobench.benchmarks.ml.ml_benchmark_template import Benchmark


class SVMBenchmark(Benchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1
    ):
        super(SVMBenchmark, self).__init__(task_id, seed, valid_size, fidelity_choice)
        self.cache_size = 200

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        # from https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/classification/libsvm_svc.p
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                "C", 0.03125, 32768, log=True, default_value=1.0
            ),
            CS.UniformFloatHyperparameter(
                "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed=None, fidelity_choice=None):
        """Fidelity space available --- specifies the fidelity dimensions

        For SVM, only a single fidelity exists, i.e., subsample fraction.
        if fidelity_choice == 0
            uses the entire data (subsample=1), reflecting the black-box setup
        else
            parameterizes the fraction of data to subsample

        """
        z_cs = CS.ConfigurationSpace(seed=seed)
        subsample_lower_bound = np.max((0.1, (0.1 or self.lower_bound_train_size)))
        if fidelity_choice == 0:
            subsample = CS.Constant('subsample', value=1)
        else:
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=subsample_lower_bound, upper=1, default_value=0.33, log=False
            )
        z_cs.add_hyperparameter(subsample)
        return z_cs

    def init_model(self, config, fidelity=None, rng=None):
        # initializing model
        rng = self.rng if rng is None else rng
        config = config.get_dictionary()
        model = SVC(
            **config,
            random_state=rng,
            cache_size=self.cache_size
        )
        return model
