"""
Changelog:
==========

0.0.1:
* First implementation of the new SVM Benchmarks.
0.0.2:
* Restructuring for consistency and to match ML Benchmark Template updates.
0.0.3:
* Adding Learning Curve support.
0.0.4:
* Extending to multi-objective query.
"""

from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter
from sklearn.svm import SVC

from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

__version__ = '0.0.4'


class SVMBenchmark(MLBenchmark):
    """ Multi-multi-fidelity SVM Benchmark
    """
    def __init__(
            self,
            task_id: int,
            valid_size: float = 0.33,
            rng: Union[np.random.RandomState, int, None] = None,
            data_path: Union[str, None] = None
    ):
        super(SVMBenchmark, self).__init__(task_id, valid_size, rng, data_path)
        self.cache_size = 1024  # in MB

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
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            SVMBenchmark._get_fidelity_choices(subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(subsample_choice: str) -> Hyperparameter:
        """Fidelity space available --- specifies the fidelity dimensions
        """
        assert subsample_choice in ['fixed', 'variable']

        fidelity = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1.0, default_value=1.0, log=False
            )
        )
        subsample = fidelity[subsample_choice]
        return subsample

    def init_model(
            self,
            config: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            rng: Union[int, np.random.RandomState, None] = None
    ):
        # initializing model
        rng = self.rng if rng is None else rng
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        model = SVC(
            **config,
            random_state=rng,
            cache_size=self.cache_size
        )
        return model

    def get_model_size(self, model: SVC) -> float:
        """ Returns the number of support vectors in the SVM model

        Parameters
        ----------
        model : SVC
            Trained SVM model.

        Returns
        -------
        float
        """
        nsupport = model.support_.shape[0]
        return nsupport


class SVMBenchmarkBB(SVMBenchmark):
    """ Black-box version of the SVMBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            # uses the entire data (subsample=1), reflecting the black-box setup
            SVMBenchmark._get_fidelity_choices(subsample_choice='fixed')
        )
        return fidelity_space


class SVMMOBenchmark(SVMBenchmark):
    def __init__(
            self,
            task_id: int,
            valid_size: float = 0.33,
            rng: Union[np.random.RandomState, int, None] = None,
            data_path: Union[str, None] = None
    ):
        super(SVMMOBenchmark, self).__init__(task_id, valid_size, rng, data_path)

    def get_objective_names(self):
        return ["loss", "inference_time"]

    def _get_multiple_objectives(self, result):
        single_obj = result['function_value']
        seeds = result['info'].keys()
        total_inference_time = sum([result['info']['val_costs']['acc']])
        avg_inference_time = total_inference_time / len(seeds)
        result['function_value'] = dict(
            loss=single_obj,
            inference_time=avg_inference_time
        )
        return result

    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            record_train: bool = False,
            get_learning_curve: bool = False,
            lc_every_k: int = 1,
            **kwargs
    ):
        result = super(SVMMOBenchmark, self).objective_function(
            configuration=configuration,
            fidelity=fidelity,
            shuffle=shuffle,
            rng=rng,
            record_train=record_train,
            get_learning_curve=get_learning_curve,
            lc_every_k=lc_every_k,
            **kwargs
        )
        result = self._get_multiple_objectives(result)
        return result

    def objective_function_test(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            record_train: bool = False,
            get_learning_curve: bool = False,
            lc_every_k: int = 1,
            **kwargs
    ):
        result = super(SVMMOBenchmark, self).objective_function_test(
            configuration=configuration,
            fidelity=fidelity,
            shuffle=shuffle,
            rng=rng,
            record_train=record_train,
            get_learning_curve=get_learning_curve,
            lc_every_k=lc_every_k,
            **kwargs
        )
        result = self._get_multiple_objectives(result)
        return result


class SVMMOBenchmarkBB(SVMBenchmark):
    """ Black-box version of the SVMBenchmark
    """
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            # uses the entire data (subsample=1), reflecting the black-box setup
            SVMBenchmark._get_fidelity_choices(subsample_choice='fixed')
        )
        return fidelity_space


# To keep the parity of the the overall design
SVMBenchmarkMF = SVMBenchmark
SVMMOBenchmarkMF = SVMMOBenchmark

__all__ = [
    'SVMBenchmark', 'SVMBenchmarkMF', 'SVMBenchmarkBB',
    'SVMMOBenchmark', 'SVMMOBenchmarkMF', 'SVMMOBenchmarkBB',
]
