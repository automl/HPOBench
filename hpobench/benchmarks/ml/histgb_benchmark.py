import numpy as np
import ConfigSpace as CS
from copy import deepcopy
from typing import Union

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class HistGBBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1,
            data_path: Union[str, None] = None
    ):
        super(HistGBBenchmark, self).__init__(task_id, seed, valid_size, fidelity_choice, data_path)
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
                'min_samples_leaf', lower=1, upper=64, default_value=1, log=True
            ),
            #TODO: fix lr value range error in map_to_config()
            CS.UniformFloatHyperparameter(
                'learning_rate', lower=1e-5, upper=1e-1, default_value=0.1, log=True
            ),
            #TODO: find best way to encode l2 reg. since log params cannot have 0 as exact bound
            # scales the regularization parameter by using it as a power of 10
            # such that the range of the parameter becomes {0, 1e-7, 1e-6, ..., 1e-1}
            # where 10 ** 0 is enforced to be 0 (no regularization)
            CS.UniformIntegerHyperparameter(
                'l2_regularization', lower=-7, upper=0, default_value=0, log=False
            )  # value of 1 indicates 0 regularization
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
                'n_estimators', lower=2, upper=100, default_value=10, log=False
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
        rng = self.rng if rng is None else rng
        config = deepcopy(config).get_dictionary()
        l2 = config.pop("l2_regularization")
        l2 = 0 if l2 == 1 else 10 ** l2
        # TODO: decide on encoding of learning rate
        #TODO: allow non-encoded categoricals?
        #TODO: early stopping set to False?
        model = HistGradientBoostingClassifier(
            **config,
            l2_regularization=l2,
            max_iter=fidelity['n_estimators'],  # a fidelity being used during initialization
            early_stopping=False,
            random_state=rng
        )
        return model
