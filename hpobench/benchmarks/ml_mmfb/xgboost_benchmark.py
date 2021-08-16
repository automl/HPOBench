from typing import Union, Tuple, Dict

import ConfigSpace as CS
import numpy as np
import xgboost as xgb
from ConfigSpace.hyperparameters import Hyperparameter

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark


class XGBoostBaseBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(XGBoostBaseBenchmark, self).__init__(task_id, rng, valid_size, data_path)

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
        raise NotImplementedError()

    @staticmethod
    def _get_fidelity_choices(n_estimators_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        assert n_estimators_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            fixed=CS.Constant('n_estimators', value=100),  # TODO: Should this be 1000 or 100?
            variable=CS.UniformIntegerHyperparameter(
                'n_estimators', lower=50, upper=2000, default_value=1000, log=False
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

    def init_model(self,
                   config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        # TODO: This seems to be wrong. (AND-condition)
        rng = rng if (rng is None and isinstance(rng, int)) else self.seed
        extra_args = dict(
            booster="gbtree",
            n_estimators=fidelity['n_estimators'],
            objective="binary:logistic",
            random_state=rng,
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


class XGBoostSearchSpace0Benchmark(XGBoostBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            XGBoostBaseBenchmark._get_fidelity_choices(n_estimators_choice='fixed', subsample_choice='fixed')
        )
        return fidelity_space


class XGBoostSearchSpace1Benchmark(XGBoostBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - ntrees
            XGBoostBaseBenchmark._get_fidelity_choices(n_estimators_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space


class XGBoostSearchSpace2Benchmark(XGBoostBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - data subsample
            XGBoostBaseBenchmark._get_fidelity_choices(n_estimators_choice='fixed', subsample_choice='variable')
        )
        return fidelity_space


class XGBoostSearchSpace3Benchmark(XGBoostBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - ntrees + data subsample
            XGBoostBaseBenchmark._get_fidelity_choices(n_estimators_choice='variable', subsample_choice='variable')
        )
        return fidelity_space


__all__ = [XGBoostSearchSpace0Benchmark, XGBoostSearchSpace1Benchmark,
           XGBoostSearchSpace2Benchmark, XGBoostSearchSpace3Benchmark]
