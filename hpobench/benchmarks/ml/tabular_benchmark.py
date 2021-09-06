"""
Changelog:
==========

0.0.1:
* First implementation of the Tabular Benchmark.
"""

from pathlib import Path
from typing import Union, List, Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.read_and_write import json as json_cs

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.dependencies.ml.ml_benchmark_template import metrics
from hpobench.util.data_manager import TabularDataManager

__version__ = '0.0.1'


class TabularBenchmark(AbstractBenchmark):

    def __init__(self,
                 model: str, task_id: int,
                 data_dir: Union[Path, str, None] = None,
                 rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        models = ['lr', 'svm', 'xgb', 'rf', 'nn']
        assert model in models, f'Parameter `model` has to be one of {models} but was {model}'

        self.task_id = task_id
        self.model = model

        self.dm = TabularDataManager(model, task_id, data_dir)
        self.table, self.metadata = self.dm.load()

        self.exp_args = self.metadata["exp_args"]
        self.config_spaces = self.metadata["config_spaces"]
        self.global_minimums = self.metadata["global_min"]

        self.original_cs = json_cs.read(self.config_spaces['x'])
        self.original_fs = json_cs.read(self.config_spaces['z'])

        super(TabularBenchmark, self).__init__(rng=rng, **kwargs)

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           seed: Union[int, None] = None,
                           metric: Union[str, None] = 'acc',
                           **kwargs) -> Dict:

        result = self._objective(configuration, fidelity, seed, metric, evaluation="val")
        return result

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function_test(self,
                                configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                seed: Union[int, None] = None,
                                metric: Union[str, None] = 'acc',
                                **kwargs) -> Dict:

        result = self._objective(configuration, fidelity, seed, metric, evaluation="test")
        return result

    # pylint: disable=arguments-differ
    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = json_cs.read(self.config_spaces['x_discrete'])
        # cs = self._preprocess_configspace(cs)
        cs.seed(seed)
        return cs

    # pylint: disable=arguments-differ
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = json_cs.read(self.config_spaces['z_discrete'])
        cs.seed(seed=seed)
        return cs

    # pylint: disable=arguments-differ
    def get_meta_information(self) -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'TabularBenchmark',
                'references': [],
                'task_id': self.task_id,
                'model': self.model,
                'original_configuration_space': self.original_cs,
                'original_fidelity_space': self.original_fs,
                }

    def _preprocess_configspace(self, config_space: CS.ConfigurationSpace) -> CS.ConfigurationSpace:
        """ Converts floats to np.float32 """
        for hp in config_space.get_hyperparameters():
            hp.sequence = tuple(np.array(hp.sequence).astype(np.float32))
            hp.default_value = np.float32(hp.default_value)
        return config_space

    def _total_number_of_configurations(self, space: str = "hyperparameters") -> int:
        """ Returns the number of unique configurations in the parameter/fidelity space
        """
        count = 1
        cs = self.configuration_space if space == "hyperparameters" else self.fidelity_space
        for hp in cs.get_hyperparameters():
            count *= len(hp.sequence)
        return count

    def _seeds_used(self) -> List:
        return self.table.seed.unique().tolist()

    def sample_hyperparameter(self, n: int = 1) -> Union[CS.Configuration, List]:
        return self.configuration_space.sample_configuration(n)

    def sample_fidelity(self, n: int = 1) -> Union[CS.Configuration, List]:
        return self.fidelity_space.sample_configuration(n)

    def get_global_min(self, metric: str = "acc"):
        """ Retrieves the minimum (1 - metric) for train, validation and test splits
        """
        assert metric in self.global_minimums.keys(), \
            "Not a valid metric: {}".format(list(self.global_minimums.keys()))
        return self.global_minimums[metric]

    def get_max_fidelity(self) -> Dict:
        max_fidelity = dict()
        for hp in self.fidelity_space.get_hyperparameters():
            max_fidelity[hp.name] = np.sort(hp.sequence)[-1]
        return max_fidelity

    def get_fidelity_range(self) -> List:
        fidelities = []
        for hp in self.fidelity_space.get_hyperparameters():
            if not isinstance(hp, CS.Constant) and len(hp.sequence) > 1:
                fidelities.append((hp.name, hp.sequence[0], hp.sequence[-1]))
        return fidelities

    def _search_dataframe(self, row_dict, df):
        # https://stackoverflow.com/a/46165056/8363967
        mask = np.array([True] * df.shape[0])
        for i, param in enumerate(df.drop("result", axis=1).columns):
            mask *= df[param].values == row_dict[param]
        idx = np.where(mask)
        assert len(idx) == 1, 'The query has resulted into mulitple matches. This should not happen. ' \
                              f'The Query was {row_dict}'
        idx = idx[0][0]
        result = df.iloc[idx]["result"]
        return result

    def _objective(
            self,
            config: Dict,
            fidelity: Dict,
            seed: Union[int, None] = None,
            metric: Union[str, None] = "acc",
            evaluation: Union[str, None] = ""
    ) -> Dict:

        metric_str = ', '.join(list(metrics.keys()))
        assert metric in list(metrics.keys()), f"metric not found among: {metric_str}"
        score_key = f"{evaluation}_scores"
        cost_key = f"{evaluation}_scores"

        key_path = dict()
        for name in self.configuration_space.get_hyperparameter_names():
            key_path[str(name)] = config[str(name)]
        for name in self.fidelity_space.get_hyperparameter_names():
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
        result = dict(function_value=float(loss), cost=costs, info=info)
        return result


__all__ = ['TabularBenchmark']
