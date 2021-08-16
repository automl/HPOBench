from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.svm import SVC

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark


class SVMBaseBenchmark(MLBenchmark):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(SVMBaseBenchmark, self).__init__(task_id, rng, valid_size, data_path)

        self.cache_size = 200

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        # https://jmlr.org/papers/volume20/18-444/18-444.pdf (Table 1)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                "C", 2**-10, 2**10, log=True, default_value=1.0
            ),
            CS.UniformFloatHyperparameter(
                "gamma", 2**-10, 2**10, log=True, default_value=0.1
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Fidelity space available --- specifies the fidelity dimensions

        For SVM, only a single fidelity exists, i.e., subsample fraction.
        if fidelity_choice == 0
            uses the entire data (subsample=1), reflecting the black-box setup
        else
            parameterize the fraction of data to subsample

        """
        raise NotImplementedError()

    @staticmethod
    def _get_fidelity_choices(subsample_choice: str) -> Hyperparameter:

        assert subsample_choice in ['fixed', 'variable']

        fidelity = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1.0, default_value=1.0, log=False
            )
        )
        subsample = fidelity[subsample_choice]

        return subsample

    def init_model(self, config: Dict, fidelity: Dict = None, rng: Union[int, np.random.RandomState, None] = None):
        # initializing model
        rng = self.rng if rng is None else rng
        config = config
        model = SVC(
            **config,
            random_state=rng,
            cache_size=self.cache_size
        )
        return model


class SVMSearchSpace0Benchmark(SVMBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            # uses the entire data (subsample=1), reflecting the black-box setup
            SVMBaseBenchmark._get_fidelity_choices(subsample_choice='fixed')
        )
        return fidelity_space


class SVMSearchSpace1Benchmark(SVMBaseBenchmark):
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            # parameterize the fraction of data to subsample
            SVMBaseBenchmark._get_fidelity_choices(subsample_choice='fixed')
        )
        return fidelity_space
