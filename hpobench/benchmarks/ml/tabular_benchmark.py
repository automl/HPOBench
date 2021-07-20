import os
import glom
import numpy as np
import ConfigSpace as CS
import pickle5 as pickle
from copy import deepcopy
from typing import Union, List, Dict
from hpobench.benchmarks.ml.ml_benchmark_template import metrics


class TabularBenchmark:
    def __init__(self, table_path: str, seed: Union[int, None] = None):
        assert os.path.isfile(table_path), "Not a valid path: {}".format(table_path)
        table = self._load_table(table_path)
        self.seed = seed if seed is not None else np.random.randint(1, 10 ** 6)
        self.rng = np.random.RandomState(self.seed)
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

    def _get_model_name(self):
        return self.exp_args["space"]

    def _total_number_of_configurations(self, space: str="hyperparameters") -> int:
        """ Returns the number of unique configurations in the parameter/fidelity space
        """
        count = 1
        cs = self.x_cs if space == "hyperparameters" else self.z_cs
        for hp in cs.get_hyperparameters():
            count *= len(hp.sequence)
        return count

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

    def get_max_fidelity(self) -> Dict:
        max_fidelity = dict()
        for hp in self.z_cs.get_hyperparameters():
            max_fidelity[hp.name] = np.sort(hp.sequence)[-1]
        return max_fidelity

    def get_fidelity_range(self):
        fidelities = []
        for hp in self.z_cs.get_hyperparameters():
            if not isinstance(hp, CS.Constant) and len(hp.sequence) > 1:
                fidelities.append((hp.name, hp.sequence[0], hp.sequence[-1]))
        return fidelities

    def _objective(
            self,
            config: CS.Configuration,
            fidelity: CS.Configuration,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc",
            eval: Union[str] = "val"
    ) -> Dict:
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
        seeds = list(val.keys())
        assert metric in list(metrics.keys()), \
            "metric not found among: {{{}}}".format(", ".join(list(metrics.keys())))
        score_key = "{}_scores".format(eval)
        cost_key = "{}_scores".format(eval)
        if seed is None:
            result = dict(function_value=0.0, cost=0.0, info=dict())
            loss = []
            costs = 0.0
            info = dict()
            for seed in seeds:
                result = deepcopy(val[seed])
                loss.append(1 - result["info"][score_key][metric])
                costs += result["info"]["model_cost"] + result["info"][cost_key][metric]
                info[seed] = result["info"]
            loss = np.mean(loss)
            result["function_value"] = loss
            result["cost"] = costs
            result["info"] = info
        else:
            assert seed in list(val.keys()), \
                "seed not found among: {{{}}}".format(", ".join([str(s) for s in seeds]))
            result = deepcopy(val[seed])
            result["function_value"] = 1 - result["info"][score_key][metric]
            result["cost"] = result["info"]["model_cost"] + result["info"][cost_key][metric]
        return result

    def objective_function(
            self,
            config: CS.Configuration,
            fidelity: CS.Configuration,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc"
    ) -> Dict:
        result = self._objective(config, fidelity, seed, metric, eval="val")
        return result

    def objective_function_test(
            self,
            config: CS.Configuration,
            fidelity: CS.Configuration,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc"
    ) -> Dict:
        result = self._objective(config, fidelity, seed, metric, eval="test")
        return result
