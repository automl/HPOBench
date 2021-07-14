import os
import glom
import numpy as np
import ConfigSpace as CS
import pickle5 as pickle
from typing import Union, List


class TabularBenchmark:
    def __init__(self, table_path: str, seed: Union[int, None]=None):
        assert os.path.isfile(table_path), "Not a valid path: {}".format(table_path)
        table = self._load_table(table_path)
        self.seed = seed if seed is not None else np.random.randint(1, 10 ** 6)
        self.exp_args = table['exp_args']
        self.config_spaces = table['config_spaces']
        self.x_cs = self.get_hyperparameter_space(seed=self.seed)
        self.z_cs = self.get_fidelity_space(seed=self.seed)
        self.table = table['data']
        self.global_minimums = table['global_min']

    def _load_table(self, path):
        with open(path, "rb") as f:
            table = pickle.load(f)
        return table

    def get_hyperparameter_space(self, seed=None, original=False):
        cs = CS.ConfigurationSpace(seed=seed)
        if original:
            _cs = self.config_spaces['x']
        _cs = self.config_spaces['x_discrete']
        for hp in _cs.get_hyperparameters():
            cs.add_hyperparameter(hp)
        return cs

    def get_fidelity_space(self, seed=None, original=False):
        cs = CS.ConfigurationSpace(seed=seed)
        if original:
            _cs = self.config_spaces['z']
        _cs = self.config_spaces['z_discrete']
        for hp in _cs.get_hyperparameters():
            cs.add_hyperparameter(hp)
        return cs

    def sample_hyperparamer(self, n: int = 1) -> Union[CS.Configuration, List]:
        return self.x_cs.sample_configuration(n)

    def sample_fidelity(self, n: int = 1) -> Union[CS.Configuration, List]:
        return self.z_cs.sample_configuration(n)

    def get_global_min(self, metric: str = "acc"):
        """ Retrieves the minimum (1 - metric) for train, validation and test splits
        """
        assert metric in self.global_minimums.keys(), \
            "Not a valid metric: {}".format(list(self.global_minimums.keys()))
        return self.global_minimums[metric]

    def objective_function(self, config, fidelity):
        self.x_cs.check_configuration(config)
        self.z_cs.check_configuration(fidelity)
        key_path = []
        for name in np.sort(self.x_cs.get_hyperparameter_names()):
            key_path.append(config[str(name)])
        for name in np.sort(self.z_cs.get_hyperparameter_names()):
            key_path.append(fidelity[str(name)])
        val = glom.glom(self.table, glom.Path(*key_path), default=None)
        if val is None:
            raise ValueError(
                "Invalid config-fidelity or not recorded in table!\n{}\n{}".format(config, fidelity)
            )
        return val
