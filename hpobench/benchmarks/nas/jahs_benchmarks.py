"""

Installation:
=============
``` pip install git+https://github.com/automl/jahs_bench_201.git ```
or along with hpobench by calling
`pip install -e .[jahs_bench_201] `

Version:
========
0.0.1:
    Initial commit using JAHS-Bench-201 version 1.0.2.
    It supports the surrogate and the tabular benchmark as MO as well as SO version.

    The MO version of this benchmark contains the following objectives:
        - 'valid-misclassification_rate'
        - 'latency'
    However there are more, which are included in the info field:
        - 'valid-acc'
        - 'valid-misclassification_rate'
        - 'latency'
        - 'FLOPS'
        - 'size_MB'
        - 'test-acc'
        - 'test-misclassification_rate'
        - 'train-acc'
        - 'train-misclassification_rate'

    The SO version returns only the valid-misclassification_rate.

    Note: Due to floating point errors, the tabular-based benchmarks might not find the correct configuration.
    We added a small fix to select (if possible) the nearest configuration with a threshold of 1e-6.
    If the configuration is not present in the tabular benchmark, it returns the empirical worst results.
    See FAILURE_VALUES
"""

import copy
import logging
import typing
from typing import Dict, Union, Tuple

import ConfigSpace
import jahs_bench as jahs_bench_code
import numpy as np

from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark, AbstractSingleObjectiveBenchmark
from hpobench.util.data_manager import JAHSDataManager

__version__ = '0.0.1'
logger = logging.getLogger('JAHSBenchmark')


class _JAHSBenchmark:

    def __init__(self, task: str, kind: str, metrics: typing.List[str] = None, **kwargs):
        assert task in ['cifar10', 'colorectal_histology', 'fashion_mnist']
        assert task in [task.value for task in jahs_bench_code.BenchmarkTasks]
        assert kind in ['surrogate', 'table', 'live']
        assert kind in [kind.value for kind in jahs_bench_code.BenchmarkTypes]

        self.task = task
        self.kind = kind
        self.metrics = metrics or self.get_objective_names()
        if 'runtime' not in metrics:
            self.metrics.append('runtime')

        # Check if we have to download the data
        self.data_manager = JAHSDataManager()
        self.data_manager.load()

        self.jahs_benchmark = jahs_bench_code.Benchmark(
            task=self.task, kind=self.kind, download=False,
            save_dir=self.data_manager.data_dir,
            metrics=[m for m in self.metrics if 'misclassification_rate' not in m],  # we compute these on the fly
        )

        self.TABULAR_METRIC_BOUNDS = {
            'cifar10': {
                'FLOPS': [0.035754, 220.11969],
                'latency': [0.00635562329065232, 119.7417876488277],
                'runtime': [20.007991552352905, 702458.5486302376],
                'size_MB': [0.004894, 1.531546],
                'train-acc': [7.26799999786377, 100.0],
                'valid-acc': [1.707999999885559, 92.92399997314453],
                'test-acc': [1.62, 92.73],
                'train-misclassification_rate': [0.0, 92.73200000213623],
                'valid-misclassification_rate': [7.076000026855468, 98.29200000011444],
                'test-misclassification_rate': [7.269999999999996, 98.38],
            },
            'colorectal_histology': {
                'FLOPS': [0.03572, 2515.2722],
                'latency': [0.006143425535297438, 798.9547640807386],
                'runtime': [1.4989218711853027, 252615.9303767681],
                'size_MB': [0.00486, 1.531416],
                'train-acc': [0.34810126582278483, 100.0],
                'valid-acc': [0.0, 97.39583333333333],
                'test-acc': [0.0, 97.1774193548387],
                'train-misclassification_rate': [0.0, 99.65189873417721],
                'valid-misclassification_rate': [2.6041666666666714, 100.0],
                'test-misclassification_rate': [2.8225806451612954, 100.0],
            },
            'fashion_mnist': {
                'FLOPS': [0.031146, 152.978058],
                'latency': [0.007562867628561484, 34.07263234384824],
                'runtime': [19.689435482025146, 638326.5186192989],
                'size_MB': [0.004822, 1.073178],
                'train-acc': [6.603174602958351, 100.0],
                'valid-acc': [0.0, 95.56613753545852],
                'test-acc': [0.0, 95.60000002615793],
                'train-misclassification_rate': [0.0, 93.39682539704165],
                'valid-misclassification_rate': [4.433862464541477, 100.0],
                'test-misclassification_rate': [4.399999973842071, 100.0],
            }}

        self.FAILURE_VALUES = {
            'cifar10': {
                'FLOPS': 0.035754,
                'latency': 119.7417876488277,
                'runtime': 702458.5486302376,
                'size_MB': 1.531546,
                'train-acc': 7.26799999786377,
                'valid-acc': 1.707999999885559,
                'test-acc': 1.62,
                'train-misclassification_rate': 100 - 7.26799999786377,
                'valid-misclassification_rate': 100 - 1.707999999885559,
                'test-misclassification_rate': 100 - 1.62,
            },
            'colorectal_histology': {
                'FLOPS': 0.03572,
                'latency': 798.9547640807386,
                'runtime': 252615.9303767681,
                'size_MB': 1.531416,
                'train-acc': 0.34810126582278483,
                'valid-acc': 0.0,
                'test-acc': 0.0,
                'train-misclassification_rate': 100 - 0.34810126582278483,
                'valid-misclassification_rate': 100 - 0.0,
                'test-misclassification_rate': 100 - 0.0,
            },
            'fashion_mnist': {
                'FLOPS': 0.031146,
                'latency': 34.07263234384824,
                'runtime': 638326.5186192989,
                'size_MB': 1.073178,
                'train-acc': 6.603174602958351,
                'valid-acc': 0.0,
                'test-acc': 0.0,
                'train-misclassification_rate': 100 - 6.603174602958351,
                'valid-misclassification_rate': 100 - 0.0,
                'test-misclassification_rate': 100 - 0.0,
            }
        }

        self.subset_metrics = ['valid-misclassification_rate', 'latency']

        super(_JAHSBenchmark, self).__init__(**kwargs)

    def normalize_metric(self, data, dataset, key="latency"):
        _min = min(self.TABULAR_METRIC_BOUNDS[dataset][key])
        _max = max(self.TABULAR_METRIC_BOUNDS[dataset][key])
        return (data - _min) / (_max - _min)

    @staticmethod
    def get_objective_names():
        return ["FLOPS", "latency", "runtime", "size_MB", "train-acc", "valid-acc", "test-acc",
                "train-misclassification_rate", "valid-misclassification_rate", "test-misclassification_rate"]

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        fidelity_space = ConfigSpace.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameter(
            ConfigSpace.UniformIntegerHyperparameter('nepochs', lower=1, upper=200, default_value=200)
        )
        return fidelity_space

    @staticmethod
    def get_meta_information() -> Dict:
        # TODO
        return {}

    def _query_benchmark(self, configuration: Union[ConfigSpace.Configuration, Dict],
                         fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                         rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Tuple[Dict, float]:
        if self.kind == 'table':
            # Some parameters cant be directly queried due to rounding issues.
            # Map these parameters (learning_rate, WeightDecay) to their closest value in the table.
            # If the difference is too large, the configuration might be not in the search space.
            # JAHSbench throws a KeyError in this case.
            subsets = self.jahs_benchmark._table_features.loc[:, ['WeightDecay', 'LearningRate']]
            subset_config = [configuration['WeightDecay'], configuration['LearningRate']]
            max_diffs = (subsets - subset_config).abs().sum(axis=1)
            index_min = max_diffs.idxmin()
            nearest_points = subsets.iloc[index_min].to_dict()

            if max_diffs.iloc[index_min] <= 1e-6:
                # We can assume that this is a floating point error.
                configuration.update(nearest_points)

            # print(subset_config)
            # print([
            #     subsets.iloc[index_min].loc['WeightDecay'],
            #     subsets.iloc[index_min].loc['LearningRate']
            # ])

        try:
            # Query the benchmark
            result = self.jahs_benchmark(configuration, nepochs=fidelity['nepochs'], full_trajectory=False)

            # The last epoch contains the interesting results for us
            if self.kind == 'table':
                _id = list(result.keys())
                result_last_epoch = result[_id[0]]
            else:
                result_last_epoch = result[fidelity['nepochs']]

        except KeyError:
            result_last_epoch = self.FAILURE_VALUES[self.task]

        _result_last_epoch = {}
        # Replace all accuracies with the misclassification rate (100 - acc)
        for k, v in result_last_epoch.items():
            _result_last_epoch[k] = v
            if 'acc' in k:
                _result_last_epoch[k.replace('acc', 'misclassification_rate')] = 100 - v

        cost = _result_last_epoch.pop('runtime')
        return _result_last_epoch, cost


class _JAHSMOBenchmark(_JAHSBenchmark, AbstractMultiObjectiveBenchmark):

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        result_last_epoch, cost = self._query_benchmark(configuration, fidelity, rng, **kwargs)
        function_value = {
            key: value
            for key, value in result_last_epoch.items()
            if key in self.subset_metrics
        }
        # function_value = {
        #     key: self.normalize_metric(data=value, dataset=self.task, key=key)
        #     for key, value in function_value.items()
        # }

        result_dict = {
            'function_value': function_value,
            'cost': cost,
            'info': copy.deepcopy(result_last_epoch)
        }

        return result_dict

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[ConfigSpace.Configuration, Dict],
                                fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity, rng, **kwargs)


class _JAHSSOBenchmark(_JAHSBenchmark, AbstractSingleObjectiveBenchmark):

    @AbstractSingleObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        result_last_epoch, cost = self._query_benchmark(configuration, fidelity, rng, **kwargs)
        function_value = {
            key: value
            for key, value in result_last_epoch.items()
            if key in self.subset_metrics
        }
        # function_value = {
        #     key: self.normalize_metric(data=value, dataset=self.task, key=key)
        #     for key, value in function_value.items()
        # }

        # Select only the validation misclassification rate!
        result_dict = {
            'function_value': function_value['valid-misclassification_rate'],
            'cost': cost,
            'info': copy.deepcopy(result_last_epoch)
        }

        return result_dict

    @AbstractSingleObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[ConfigSpace.Configuration, Dict],
                                fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity, rng, **kwargs)


class _SurrogateSearchSpace:
    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        from jahs_bench.lib.core.configspace import joint_config_space
        joint_config_space.seed(seed)
        return joint_config_space


class _TabularSearchSpace:
    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        # from jahs_bench.tabular.search_space.configspace import joint_config_space
        from jahs_bench.lib.core.configspace import joint_config_space
        joint_config_space.seed(seed)
        return joint_config_space


# ######################### Single Objective - Surrogate ###############################################################
class JAHSSOCifar10SurrogateBenchmark(_SurrogateSearchSpace, _JAHSSOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSSOCifar10SurrogateBenchmark, self).__init__(
            task='cifar10', rng=rng, kind='surrogate', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSSOColorectalHistologySurrogateBenchmark(_SurrogateSearchSpace, _JAHSSOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSSOColorectalHistologySurrogateBenchmark, self).__init__(
            task='colorectal_histology', rng=rng, metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSSOFashionMNISTSurrogateBenchmark(_SurrogateSearchSpace, _JAHSSOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSSOFashionMNISTSurrogateBenchmark, self).__init__(
            task='fashion_mnist', rng=rng, metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )
# ######################### Single Objective - Surrogate ###############################################################


# ######################### Single Objective - Tabular ###############################################################
class JAHSSOCifar10TabularBenchmark(_TabularSearchSpace, _JAHSSOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSSOCifar10TabularBenchmark, self).__init__(
            task='cifar10', rng=rng, kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSSOColorectalHistologyTabularBenchmark(_TabularSearchSpace, _JAHSSOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSSOColorectalHistologyTabularBenchmark, self).__init__(
            task='colorectal_histology', rng=rng, kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSSOFashionMNISTTabularBenchmark(_TabularSearchSpace, _JAHSSOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSSOFashionMNISTTabularBenchmark, self).__init__(
            task='fashion_mnist', rng=rng, kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )
# ######################### Single Objective - Tabular ###############################################################


# ######################### Multi Objective - Surrogate ###############################################################
class JAHSMOCifar10SurrogateBenchmark(_SurrogateSearchSpace, _JAHSMOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOCifar10SurrogateBenchmark, self).__init__(
            task='cifar10', rng=rng, kind='surrogate', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSMOColorectalHistologySurrogateBenchmark(_SurrogateSearchSpace, _JAHSMOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOColorectalHistologySurrogateBenchmark, self).__init__(
            task='colorectal_histology', rng=rng, metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSMOFashionMNISTSurrogateBenchmark(_SurrogateSearchSpace, _JAHSMOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOFashionMNISTSurrogateBenchmark, self).__init__(
            task='fashion_mnist', rng=rng, metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )
# ######################### Multi Objective - Surrogate ###############################################################


# ######################### Multi Objective - Tabular ###############################################################
class JAHSMOCifar10TabularBenchmark(_TabularSearchSpace, _JAHSMOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOCifar10TabularBenchmark, self).__init__(
            task='cifar10', rng=rng, kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSMOColorectalHistologyTabularBenchmark(_TabularSearchSpace, _JAHSMOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOColorectalHistologyTabularBenchmark, self).__init__(
            task='colorectal_histology', rng=rng, kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )


class JAHSMOFashionMNISTTabularBenchmark(_TabularSearchSpace, _JAHSMOBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOFashionMNISTTabularBenchmark, self).__init__(
            task='fashion_mnist', rng=rng, kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs
        )
# ######################### Multi Objective - Tabular ###############################################################


__all__ = [
    "JAHSSOCifar10SurrogateBenchmark",
    "JAHSSOColorectalHistologySurrogateBenchmark",
    "JAHSSOFashionMNISTSurrogateBenchmark",

    "JAHSSOCifar10TabularBenchmark",
    "JAHSSOColorectalHistologyTabularBenchmark",
    "JAHSSOFashionMNISTTabularBenchmark",

    "JAHSMOCifar10SurrogateBenchmark",
    "JAHSMOColorectalHistologySurrogateBenchmark",
    "JAHSMOFashionMNISTSurrogateBenchmark",

    "JAHSMOCifar10TabularBenchmark",
    "JAHSMOColorectalHistologyTabularBenchmark",
    "JAHSMOFashionMNISTTabularBenchmark",
]
