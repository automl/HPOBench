import numpy as np
import ConfigSpace as CS
from typing import Union, Dict

import xgboost as xgb

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class XGBoostBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1,
            data_path: Union[str, None] = None
    ):
        super(XGBoostBenchmark, self).__init__(
            task_id, seed, valid_size, fidelity_choice, data_path
        )
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
        fidelity1 = dict(
            fixed=CS.Constant('n_estimators', value=100),
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
        if fidelity_choice == 0:
            # black-box setting (full fidelity)
            ntrees = fidelity1["fixed"]
            subsample = fidelity2["fixed"]
        elif fidelity_choice == 1:
            # gray-box setting (multi-fidelity) - ntrees
            ntrees = fidelity1["variable"]
            subsample = fidelity2["fixed"]
        elif fidelity_choice == 2:
            # gray-box setting (multi-fidelity) - data subsample
            ntrees = fidelity1["fixed"]
            subsample = fidelity2["variable"]
        else:
            # gray-box setting (multi-multi-fidelity) - ntrees + data subsample
            ntrees = fidelity1["variable"]
            subsample = fidelity2["variable"]
        z_cs.add_hyperparameters([ntrees, subsample])
        return z_cs

    def init_model(self, config, fidelity=None, rng=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
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
            **config.get_dictionary(),
            **extra_args
        )
        return model
