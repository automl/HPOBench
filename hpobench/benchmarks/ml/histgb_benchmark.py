"""
Changelog:
==========

0.0.1:
* First implementation of the HistGB Benchmarks.
"""

from typing import Union, Tuple, Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

__version__ = '0.0.1'


class HistGBBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(HistGBBenchmark, self).__init__(task_id, rng, valid_size, data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters"""
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'max_depth', lower=6, upper=30, default_value=6, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'max_leaf_nodes', lower=2, upper=64, default_value=32, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate', lower=2**-10, upper=1, default_value=0.1, log=True
            ),
            CS.UniformFloatHyperparameter(
                'l2_regularization', lower=2**-10, upper=2**10, default_value=0.1, log=True
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - ntrees + data subsample
            HistGBBenchmark._get_fidelity_choices(ntrees_choice='variable', subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(ntrees_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        assert ntrees_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            # TODO: this value was 100 in the original code. Please check if 100 or 1000.
            fixed=CS.Constant('n_estimators', value=1000),
            variable=CS.UniformIntegerHyperparameter(
                'n_estimators', lower=100, upper=1000, default_value=1000, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        ntrees = fidelity1[ntrees_choice]
        subsample = fidelity2[subsample_choice]
        return ntrees, subsample

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        model = HistGradientBoostingClassifier(
            **config,
            max_iter=fidelity['n_estimators'],  # a fidelity being used during initialization
            early_stopping=False,
            random_state=rng
        )
        return model


class HistGBBenchmarkBB(HistGBBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            HistGBBenchmark._get_fidelity_choices(ntrees_choice='fixed', subsample_choice='fixed')
        )
        return fidelity_space


class HistGBBenchmarkMF(HistGBBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - ntrees
            HistGBBenchmark._get_fidelity_choices(ntrees_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space


__all__ = ['HistGBBenchmark', 'HistGBBenchmarkBB', 'HistGBBenchmarkMF']
