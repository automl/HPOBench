"""
Changelog:
==========

0.0.1:
* First implementation of the new XGB Benchmarks.
0.0.2:
* Restructuring for consistency and to match ML Benchmark Template updates.
0.0.3:
* Adding Learning Curve support.
"""

from typing import Union, Tuple, Dict

import ConfigSpace as CS
import numpy as np
import xgboost as xgb
from ConfigSpace.hyperparameters import Hyperparameter

from hpobench.util.rng_helper import get_rng
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

__version__ = '0.0.3'


class XGBoostBenchmark(MLBenchmark):
    """ Multi-multi-fidelity XGBoost Benchmark
    """
    def __init__(
            self,
            task_id: int,
            valid_size: float = 0.33,
            rng: Union[np.random.RandomState, int, None] = None,
            data_path: Union[str, None] = None
    ):
        super(XGBoostBenchmark, self).__init__(task_id, valid_size, rng, data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                'eta', lower=2**-10, upper=1., default_value=0.3, log=True
            ),  # learning rate
            CS.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=50, default_value=10, log=True
            ),
            CS.UniformFloatHyperparameter(
                'colsample_bytree', lower=0.1, upper=1., default_value=1., log=False
            ),
            CS.UniformFloatHyperparameter(
                'reg_lambda', lower=2**-10, upper=2**10, default_value=1, log=True
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - ntrees + data subsample
            XGBoostBenchmark._get_fidelity_choices(
                n_estimators_choice='variable', subsample_choice='variable'
            )
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(
            n_estimators_choice: str, subsample_choice: str
    ) -> Tuple[Hyperparameter, Hyperparameter]:

        assert n_estimators_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            fixed=CS.Constant('n_estimators', value=2000),
            variable=CS.UniformIntegerHyperparameter(
                'n_estimators', lower=50, upper=2000, default_value=2000, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        n_estimators = fidelity1[n_estimators_choice]
        subsample = fidelity2[subsample_choice]
        return n_estimators, subsample

    def init_model(
            self,
            config: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            rng: Union[int, np.random.RandomState, None] = None
    ):
        # initializing model
        rng = self.rng if rng is None else get_rng(rng)
        # xgb.XGBClassifier when trainied using the scikit-learn API of `fit`, requires
        # random_state to be an integer and doesn't accept a RandomState
        seed = rng.randint(1, 10**6)

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()
        extra_args = dict(
            booster="gbtree",
            n_estimators=fidelity['n_estimators'],
            objective="binary:logistic",
            random_state=seed,
            subsample=1
        )
        if self.n_classes > 2:
            extra_args["objective"] = "multi:softmax"
            extra_args.update({"num_class": self.n_classes})

        model = xgb.XGBClassifier(
            **config,
            **extra_args
        )
        return model

    def get_model_size(self, model: xgb.XGBClassifier) -> float:
        """ Returns the total number of decision nodes in the sequence of Gradient Boosted trees

        Parameters
        ----------
        model : xgb.XGBClassifier
            Trained XGB model.

        Returns
        -------
        float
        """
        nodes = model.get_booster().trees_to_dataframe().shape[0]
        return nodes


class XGBoostBenchmarkBB(XGBoostBenchmark):
    """ Black-box version of the XGBoostBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            XGBoostBenchmark._get_fidelity_choices(
                n_estimators_choice='fixed', subsample_choice='fixed'
            )
        )
        return fidelity_space


class XGBoostBenchmarkMF(XGBoostBenchmark):
    """ Multi-fidelity version of the XGBoostBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - ntrees
            XGBoostBenchmark._get_fidelity_choices(
                n_estimators_choice='variable', subsample_choice='fixed'
            )
        )
        return fidelity_space


__all__ = ['XGBoostBenchmarkBB', 'XGBoostBenchmarkMF', 'XGBoostBenchmark']
