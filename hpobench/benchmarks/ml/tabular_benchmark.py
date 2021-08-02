import os
import json
import numpy as np
import pandas as pd
import ConfigSpace as CS
from ConfigSpace.read_and_write import json as json_cs
from typing import Union, List, Dict

from hpobench.benchmarks.ml.ml_benchmark_template import metrics


class TabularBenchmark:
    def __init__(self, path: str, model: str, task_id: int, seed: Union[int, None] = None):
        assert os.path.isdir(path), "Not a valid path: {}".format(path)
        self.data_path = os.path.join(path, "{}_{}_data.parquet.gzip".format(model, task_id))
        assert os.path.isfile(self.data_path)
        self.metadata_path = os.path.join(path, "{}_{}_metadata.json".format(model, task_id))
        assert os.path.isfile(self.metadata_path)

        self.seed = seed if seed is not None else np.random.randint(1, 10 ** 6)
        self.rng = np.random.RandomState(self.seed)
        self.table = self._load_parquet(self.data_path)
        self.metadata = self._load_json(self.metadata_path)
        self.exp_args = self.metadata["exp_args"]
        self.config_spaces = self.metadata["config_spaces"]
        self.global_minimums = self.metadata["global_min"]
        self.x_cs = self.get_hyperparameter_space(seed=self.seed)
        self.z_cs = self.get_fidelity_space(seed=self.seed)

    def _load_parquet(self, path):
        data = pd.read_parquet(path)
        return data

    def _load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def _preprocess_configspace(self, config_space):
        """ Converts floats to np.float32 """
        for hp in config_space.get_hyperparameters():
            hp.sequence = tuple(np.array(hp.sequence).astype(np.float32))
            hp.default_value = np.float32(hp.default_value)
        return config_space

    def _total_number_of_configurations(self, space: str="hyperparameters") -> int:
        """ Returns the number of unique configurations in the parameter/fidelity space
        """
        count = 1
        cs = self.x_cs if space == "hyperparameters" else self.z_cs
        for hp in cs.get_hyperparameters():
            count *= len(hp.sequence)
        return count

    def _seeds_used(self):
        return self.table.seed.unique().tolist()

    def get_hyperparameter_space(self, seed=None, original=False):
        cs = CS.ConfigurationSpace(seed=seed)
        load_name = "x" if original else "x_discrete"
        _cs = json_cs.read(self.config_spaces[load_name])
        for hp in _cs.get_hyperparameters():
            cs.add_hyperparameter(hp)
        if not original:
            cs = self._preprocess_configspace(cs)
        return cs

    def get_fidelity_space(self, seed=None, original=False):
        cs = CS.ConfigurationSpace(seed=seed)
        load_name = "z" if original else "z_discrete"
        _cs = json_cs.read(self.config_spaces[load_name])
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

    def _search_dataframe(self, row_dict, df):
        # https://stackoverflow.com/a/46165056/8363967
        mask = np.array([True] * df.shape[0])
        for i, param in enumerate(df.drop("result", axis=1).columns):
            mask *= df[param].values == row_dict[param]
        idx = np.where(mask)
        if len(idx) != 1:
            return None
        idx = idx[0][0]
        result = df.iloc[idx]["result"]
        return result

    def _objective(
            self,
            config: CS.Configuration,
            fidelity: CS.Configuration,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc",
            evaluation: Union[str] = ""
    ) -> Dict:
        self.x_cs.check_configuration(config)
        self.z_cs.check_configuration(fidelity)
        assert metric in list(metrics.keys()), \
            "metric not found among: {{{}}}".format(", ".join(list(metrics.keys())))
        score_key = "{}_scores".format(evaluation)
        cost_key = "{}_scores".format(evaluation)

        key_path = dict()
        for name in np.sort(self.x_cs.get_hyperparameter_names()):
            key_path[str(name)] = config[str(name)]
        for name in np.sort(self.z_cs.get_hyperparameter_names()):
            key_path[str(name)] = fidelity[str(name)]

        if seed is not None:
            assert seed in self._seeds_used()
            seeds = [seed]
        else:
            seeds = self._seeds_used()

        loss = []
        costs = 0.0
        info = dict()
        for seed in seeds:
            key_path["seed"] = seed
            res = self._search_dataframe(key_path, self.table)
            loss.append(1 - res["info"][score_key][metric])
            costs += res["info"]["model_cost"] + res["info"][cost_key][metric]
            info[seed] = res["info"]
            key_path.pop("seed")
        loss = np.mean(loss)
        result = dict(function_value=loss, cost=costs, info=info)
        return result

    def objective_function(
            self,
            config: CS.Configuration,
            fidelity: CS.Configuration,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc"
    ) -> Dict:
        result = self._objective(config, fidelity, seed, metric, evaluation="val")
        return result

    def objective_function_test(
            self,
            config: CS.Configuration,
            fidelity: CS.Configuration,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc"
    ) -> Dict:
        result = self._objective(config, fidelity, seed, metric, evaluation="test")
        return result
