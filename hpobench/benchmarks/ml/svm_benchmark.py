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

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class SVMBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1,
            data_path: Union[str, None] = None
    ):
        super(SVMBenchmark, self).__init__(task_id, seed, valid_size, fidelity_choice, data_path)
        self.cache_size = 200

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        # https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf (Section 3.2)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                "C", 2**-5, 2**15, log=True, default_value=1.0
            ),
            CS.UniformFloatHyperparameter(
                "gamma", 2**-15, 2**3, log=True, default_value=0.1
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

        if fidelity_choice == 0:
            subsample = CS.Constant('subsample', value=1)
        else:
            # TODO: dynamically adapt based on 1/512 and lower_bound_train_size and set log=True
            lower = 0.1
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=lower, upper=1, default_value=0.33, log=False
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
