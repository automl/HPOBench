"""
Changelog:
==========

0.0.1:
* First implementation of the Surrogate Benchmark.
"""

import pickle
from pathlib import Path
from typing import Union, List, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark
from hpobench.benchmarks.ml import LRBenchmarkMF, SVMBenchmarkMF, RandomForestBenchmarkMF, \
    XGBoostBenchmarkMF, NNBenchmarkMF

__version__ = '0.0.1'


class SurrogateBenchmark(MLBenchmark):
    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, None] = None,
    ):
        super(SurrogateBenchmark, self).__init__(task_id, rng, valid_size, data_path)

    def _load_model(self, surr_path):
        with open(surr_path / "surr_loss_{}_{}.pkl".format(self.model, self.task_id), "rb") as f:
            loss_model = pickle.load(f)
        with open(surr_path / "surr_cost_{}_{}.pkl".format(self.model, self.task_id), "rb") as f:
            cost_model = pickle.load(f)
        return loss_model, cost_model

    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            **kwargs
    ) -> Dict:
        if isinstance(configuration, CS.Configuration):
            configuration = configuration.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()
        x = list(configuration.values()) + list(fidelity.values())
        loss = self.loss_model.predict([x])[0]
        cost = self.cost_model.predict([x])[0]
        result = dict(
            function_value=loss,
            cost=cost,
            info=dict()
        )
        return result


class LRSurrogateBenchmark(LRBenchmarkMF, SurrogateBenchmark):
    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, None] = None,
            surr_path:  Union[str, None] = None
    ):
        super(LRSurrogateBenchmark, self).__init__(task_id, rng, valid_size, data_path)
        self.model = "lr"
        self.task_id = task_id
        surr_path = Path(surr_path)
        self.loss_model, self.cost_model = self._load_model(surr_path)


class SVMSurrogateBenchmark(SVMBenchmarkMF, SurrogateBenchmark):
    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, None] = None,
            surr_path:  Union[str, None] = None
    ):
        super(SVMSurrogateBenchmark, self).__init__(task_id, rng, valid_size, data_path)
        self.model = "svm"
        self.task_id = task_id
        surr_path = Path(surr_path)
        self.loss_model, self.cost_model = self._load_model(surr_path)


class RFSurrogateBenchmark(RandomForestBenchmarkMF, SurrogateBenchmark):
    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, None] = None,
            surr_path:  Union[str, None] = None
    ):
        super(RFSurrogateBenchmark, self).__init__(task_id, rng, valid_size, data_path)
        self.model = "rf"
        self.task_id = task_id
        surr_path = Path(surr_path)
        self.loss_model, self.cost_model = self._load_model(surr_path)


class XGBSurrogateBenchmark(XGBoostBenchmarkMF, SurrogateBenchmark):
    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, None] = None,
            surr_path:  Union[str, None] = None
    ):
        super(XGBSurrogateBenchmark, self).__init__(task_id, rng, valid_size, data_path)
        self.model = "xgb"
        self.task_id = task_id
        surr_path = Path(surr_path)
        self.loss_model, self.cost_model = self._load_model(surr_path)


class NNSurrogateBenchmark(NNBenchmarkMF, SurrogateBenchmark):
    def __init__(
            self,
            task_id: int,
            rng: Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            data_path: Union[str, None] = None,
            surr_path:  Union[str, None] = None
    ):
        super(NNSurrogateBenchmark, self).__init__(task_id, rng, valid_size, data_path)
        self.model = "nn"
        self.task_id = task_id
        surr_path = Path(surr_path)
        self.loss_model, self.cost_model = self._load_model(surr_path)
