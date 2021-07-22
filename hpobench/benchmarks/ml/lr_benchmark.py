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
        self.cache_size = 500

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
        if fidelity_choice == 0:
            # black-box setting (full fidelity)
            iter = fidelity1["fixed"]
            subsample = fidelity2["fixed"]
        elif fidelity_choice == 1:
            # gray-box setting (multi-fidelity) - iterations
            iter = fidelity1["variable"]
            subsample = fidelity2["fixed"]
        elif fidelity_choice == 2:
            # gray-box setting (multi-fidelity) - data subsample
            iter = fidelity1["fixed"]
            subsample = fidelity2["variable"]
        else:
            # gray-box setting (multi-multi-fidelity) - iterations + data subsample
            iter = fidelity1["variable"]
            subsample = fidelity2["variable"]
        z_cs.add_hyperparameters([iter, subsample])
        return z_cs

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
