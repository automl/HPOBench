import numpy as np
import ConfigSpace as CS
from copy import deepcopy
from typing import Union, Tuple
from sklearn.neural_network import MLPClassifier

from hpobench.benchmarks.ml.ml_benchmark_template import MLBenchmark


class NNBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1,
            data_path: Union[str, None] = None
    ):
        super(NNBenchmark, self).__init__(
            task_id, seed, valid_size, fidelity_choice, data_path
        )
        # fixing layers in the architecture
        self.n_layers = 5
        pass

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.CategoricalHyperparameter(
                'shape', default_value="funnel",
                choices=["funnel", "long_funnel", "rhombus", "diamond", "hexagon",
                         "brick", "triangle", "stairs"]
            ),
            CS.OrdinalHyperparameter(
                'max_hidden_dim', sequence=[64, 128, 256, 512, 1024], default_value=128
            ),
            CS.UniformFloatHyperparameter(
                'alpha', lower=10**-5, upper=10**4, default_value=10**-3, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'batch_size', lower=4, upper=256, default_value=32, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate_init', lower=2**-10, upper=1, default_value=0.3, log=True
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed=None, fidelity_choice=1):
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
        z_cs = CS.ConfigurationSpace(seed=seed)
        fidelity1 = dict(
            fixed=CS.Constant('iter', value=100),
            variable=CS.UniformIntegerHyperparameter(
                'iter', lower=3, upper=150, default_value=30, log=False
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
            iter = fidelity1["fixed"]
            subsample = fidelity2["fixed"]
        elif fidelity_choice == 1:
            # gray-box setting (multi-fidelity) - epochs/iteration
            iter = fidelity1["variable"]
            subsample = fidelity2["fixed"]
        elif fidelity_choice == 2:
            # gray-box setting (multi-fidelity) - data subsample
            iter = fidelity1["fixed"]
            subsample = fidelity2["variable"]
        else:
            # gray-box setting (multi-multi-fidelity) - epochs + data subsample
            iter = fidelity1["variable"]
            subsample = fidelity2["variable"]
        z_cs.add_hyperparameters([iter, subsample])
        return z_cs

    def _get_architecture(self, shape: str, max_hidden_size: int) -> Tuple:
        # https://mikkokotila.github.io/slate/#shapes
        arch = []
        if shape == "funnel":
            for i in range(self.n_layers):
                arch.append(max_hidden_size)
                max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
        elif shape == "long_funnel":
            brick_arch_len = np.ceil(self.n_layers / 2).astype(int)
            for i in range(brick_arch_len):
                arch.append(max_hidden_size)
            for i in range(self.n_layers - brick_arch_len):
                max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
                arch.append(max_hidden_size)
        elif shape == "rhombus":
            arch.append(max_hidden_size)
            rhombus_len = self.n_layers // 2
            _arch = []
            for i in range(rhombus_len):
                max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
                _arch.append(max_hidden_size)
            arch = np.flip(_arch).tolist() + arch + _arch
        elif shape == "diamond":
            # open rhombus
            arch.append(max_hidden_size)
            rhombus_len = self.n_layers // 2
            second_max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
            _arch = []
            for i in range(rhombus_len):
                max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
                _arch.append(max_hidden_size)
            arch = [second_max_hidden_size] * rhombus_len + arch + _arch
        elif shape == "hexagon":
            if self.n_layers % 2 == 0:
                arch.append(max_hidden_size)
            half_len = np.ceil(self.n_layers / 2).astype(int)
            _arch = []
            for i in range(half_len):
                _arch.append(max_hidden_size)
                max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
            arch = _arch[::-1] + arch + _arch[:-1]
        elif shape == "triangle":
            # reverse funnel
            for i in range(self.n_layers):
                arch.append(max_hidden_size)
                max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
            arch = arch[::-1]
        elif shape == "stairs":
            for i in range(1, self.n_layers+1):
                arch.append(max_hidden_size)
                if i % 2 == 0 or self.n_layers < 4:
                    max_hidden_size = np.ceil(max_hidden_size / 2).astype(int)
        else:
            # default to brick design
            arch = tuple([max_hidden_size] * self.n_layers)
        arch = tuple(arch)
        return arch

    def init_model(self, config, fidelity=None, rng=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng
        config = deepcopy(config.get_dictionary())
        shape = config["shape"]
        max_hidden_dim = config["max_hidden_dim"]
        config.pop("shape")
        config.pop("max_hidden_dim")
        model = MLPClassifier(
            **config,
            hidden_layer_sizes=self._get_architecture(shape, max_hidden_dim),
            activation="relu",
            solver="sgd",
            learning_rate="invscaling",
            momentum=0.9,
            max_iter=fidelity['iter'],  # a fidelity being used during initialization
            random_state=rng
        )
        return model
