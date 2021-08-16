from copy import deepcopy
from typing import Union, Tuple

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.neural_network import MLPClassifier

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark


class NNBaseBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: Union[int, None] = None,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(NNBaseBenchmark, self).__init__(task_id, rng, valid_size, data_path)

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'depth', default_value=3, lower=1, upper=3, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'width', default_value=64, lower=16, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'batch_size', lower=4, upper=256, default_value=32, log=True
            ),
            CS.UniformFloatHyperparameter(
                'alpha', lower=10**-8, upper=1, default_value=10**-3, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Fidelity space available --- specifies the fidelity dimensions

        If fidelity_choice is 0
            Fidelity space is the maximal fidelity, akin to a black-box function
        If fidelity_choice is 1
            Fidelity space is a single fidelity, in this case the number of epochs (max_iter)
        If fidelity_choice is 2
            Fidelity space is a single fidelity, in this case the fraction of dataset (subsample)
        If fidelity_choice is >2
            Fidelity space is multi-multi fidelity, all possible fidelities
        """
        raise NotImplementedError()

    @staticmethod
    def _get_fidelity_choices(iter_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        fidelity1 = dict(
            fixed=CS.Constant('iter', value=100),
            variable=CS.UniformIntegerHyperparameter(
                'iter', lower=3, upper=243, default_value=243, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        iter = fidelity1[iter_choice]
        subsample = fidelity2[subsample_choice]
        return iter, subsample

    def init_model(self, config, fidelity=None, rng=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        config = deepcopy(config.get_dictionary())
        depth = config["depth"]
        width = config["width"]
        config.pop("depth")
        config.pop("width")
        hidden_layers = [width] * depth
        model = MLPClassifier(
            **config,
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=fidelity['iter'],  # a fidelity being used during initialization
            random_state=rng
        )
        return model


class NNSearchSpace0Benchmark(NNBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # black-box setting (full fidelity)
            NNSearchSpace0Benchmark._get_fidelity_choices(iter_choice='fixed', subsample_choice='fixed')
        )
        return fidelity_space


class NNSearchSpace1Benchmark(NNBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - iterations
            NNSearchSpace1Benchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='fixed')
        )
        return fidelity_space


class NNSearchSpace2Benchmark(NNBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-fidelity) - subsample
            NNSearchSpace2Benchmark._get_fidelity_choices(iter_choice='fixed', subsample_choice='variable')
        )
        return fidelity_space


class NNSearchSpace3Benchmark(NNBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - iterations + data subsample
            NNSearchSpace3Benchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='variable')
        )
        return fidelity_space


__all__ = [NNSearchSpace0Benchmark, NNSearchSpace1Benchmark,
           NNSearchSpace2Benchmark, NNSearchSpace3Benchmark]
