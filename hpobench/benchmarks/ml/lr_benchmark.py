import ConfigSpace as CS
from typing import Union, List, Dict

from sklearn.linear_model import SGDClassifier

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class LRBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1,
            data_path: Union[str, None] = None
    ):
        super(LRBenchmark, self).__init__(task_id, seed, valid_size, fidelity_choice, data_path)
        self.cache_size = 200

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(
                "alpha", 10**-5, 10**4, log=True, default_value=1.0
            ),
            CS.UniformFloatHyperparameter(
                "eta0", 2**-10, 1, log=True, default_value=0.3
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
            iter = CS.Constant('iter', value=1000)
            subsample = CS.Constant('subsample', value=1)
        elif fidelity_choice == 1:
            iter = CS.UniformIntegerHyperparameter(
                'iter', lower=100, upper=10000, default_value=100, log=False
            )
            subsample = CS.Constant('subsample', value=1)
        elif fidelity_choice == 2:
            iter = CS.Constant('iter', value=1000)
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        else:
            iter = CS.UniformIntegerHyperparameter(
                'iter', lower=100, upper=10000, default_value=100, log=False
            )
            subsample = CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        z_cs.add_hyperparameters([iter, subsample])
        return z_cs

    def init_model(self, config, fidelity=None, rng=None):
        # initializing model
        rng = self.rng if rng is None else rng
        config = config.get_dictionary()
        model = SGDClassifier(
            **config,
            loss="log",
            max_iter=fidelity["iter"],
            learning_rate="invscaling",
            random_state=rng
        )
        return model
