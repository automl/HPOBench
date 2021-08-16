from typing import Union, Tuple

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.linear_model import SGDClassifier

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark


class LRBaseBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: Union[int, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):

        super(LRBaseBenchmark, self).__init__(task_id, rng, valid_size, data_path)
        self.cache_size = 500  # TODO: Do we need this?

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                "alpha", 1e-5, 1, log=True, default_value=1e-3
            ),
            CS.UniformFloatHyperparameter(
                "eta0", 1e-5, 1, log=True, default_value=1e-2
            )
        ])
        return cs

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def _get_fidelity_choices(iter_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:
        """Fidelity space available --- specifies the fidelity dimensions

        For SVM, only a single fidelity exists, i.e., subsample fraction.
        if fidelity_choice == 0
            uses the entire data (subsample=1), reflecting the black-box setup
        else
            parameterizes the fraction of data to subsample

        """

        assert iter_choice in ['fixed', 'variable']
        assert subsample_choice in ['fixed', 'variable']

        fidelity1 = dict(
            fixed=CS.Constant('iter', value=1000),
            variable=CS.UniformIntegerHyperparameter(
                'iter', lower=10, upper=1000, default_value=1000, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1.0),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1.0, default_value=1.0, log=False
            )
        )

        iter = fidelity1[iter_choice]
        subsample = fidelity2[subsample_choice]
        return iter, subsample

    def init_model(self, config, fidelity=None, rng=None):
        # initializing model
        rng = self.rng if rng is None else rng
        # https://scikit-learn.org/stable/modules/sgd.html
        model = SGDClassifier(
            **config.get_dictionary(),
            loss="log",  # performs Logistic Regression
            max_iter=fidelity["iter"],
            learning_rate="adaptive",
            tol=None,
            random_state=rng,
        )
        return model


class LRSearchSpace0Benchmark(LRBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            LRBaseBenchmark._get_fidelity_choices(iter_choice='fixed', subsample_choice='fixed')
        )
        return fidelity_space


class LRSearchSpace1Benchmark(LRBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - iterations
            LRBaseBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space


class LRSearchSpace2Benchmark(LRBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - data subsample
            LRBaseBenchmark._get_fidelity_choices(iter_choice='fixed', subsample_choice='variable')
        )
        return fidelity_space


class LRSearchSpace3Benchmark(LRBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - iterations + data subsample
            LRBaseBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='variable')
        )
        return fidelity_space


__all__ = [LRSearchSpace0Benchmark, LRSearchSpace1Benchmark,
           LRSearchSpace2Benchmark, LRSearchSpace3Benchmark]
