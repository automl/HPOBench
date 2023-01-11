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
"""

import copy
import logging
import tarfile
import typing
from pathlib import Path
from typing import Dict, Union, Tuple

import ConfigSpace
import jahs_bench as jahs_bench_code
import numpy as np
import requests
from oslo_concurrency import lockutils

from hpobench import config_file
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark, AbstractSingleObjectiveBenchmark

__version__ = '0.0.1'
logger = logging.getLogger('JAHSBenchmark')


class JAHSDataManager:

    def __init__(self):
        self.data_dir = config_file.data_dir / 'jahs_data'
        self.surrogate_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/" \
                             "v1.1.0/assembled_surrogates.tar"
        self.metric_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/" \
                          "v1.1.0/metric_data.tar"

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{config_file.cache_dir}/lock_download_file_jahs', delay=0.05)
    def _load_file(self, data_dir):
        data_dir = Path(data_dir)
        surrogate_file = 'assembled_surrogates.tar'
        metric_file = 'metric_data.tar'
        if not (data_dir / f'{surrogate_file}_done.FLAG').exists():
            logger.info(f'File {surrogate_file} does not exist in {data_dir}. Start downloading.')
            self.download_and_extract_url(self.surrogate_url, data_dir, filename=surrogate_file)
        else:
            logger.info(f'File {surrogate_file} already exists. Skip downloading.')

        if not (data_dir / f'{metric_file}_done.FLAG').exists():
            logger.info(f'File {metric_file} does not exist in {data_dir}. Start downloading.')
            self.download_and_extract_url(self.metric_url, data_dir, filename=metric_file)
        else:
            logger.info(f'File {metric_file} already exists. Skip downloading.')

    def load(self):
        self._load_file(self.data_dir)

    @staticmethod
    def download_and_extract_url(url, save_dir, filename):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_tar_file = save_dir / filename
        finish_flag = save_dir / f'{filename}_done.FLAG'

        logger.info(f"Starting download of {url}, this might take a while.")
        if not save_tar_file.exists():
            with requests.get(url, stream=True) as response:
                with open(save_tar_file, 'wb') as f:
                    f.write(response.raw.read())
        else:
            logger.info(f'File: {save_tar_file} does already exist. Skip downloading!')

        logger.info("Download finished, extracting now")
        with tarfile.open(save_tar_file, 'r') as f:
            f.extractall(path=save_dir)

        if save_tar_file.name == 'assembled_surrogates.tar':
            from shutil import move
            _dir = save_dir / 'assembled_surrogates'
            _dir.mkdir(exist_ok=True, parents=True)
            for dir_name in ['cifar10', 'colorectal_histology', 'fashion_mnist']:
                _old_dir = save_dir / dir_name
                _new_dir = _dir / dir_name
                move(_old_dir, _new_dir)

        logger.info("Done extracting")

        finish_flag.touch()


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
        super(_JAHSBenchmark, self).__init__(**kwargs)

        self.METRIC_BOUNDS = {
            "latency": {
                "cifar10": [0.00635562329065232, 114.1251799692699],
                "colorectal_histology": [0.0063284998354704485, 798.9547640807386],
                "fashion_mnist": [0.007562867628561484, 9.461364439356307],
            },
            "valid-acc": [0, 100],
            "valid-misclassification_rate": [0, 100],
        }

        self.subset_metrics = ['valid-misclassification_rate', 'latency']

    def normalize_metric(self, data, dataset, key="latency"):
        if isinstance(self.METRIC_BOUNDS[key], dict):
            _min = min(self.METRIC_BOUNDS[key][dataset])
            _max = max(self.METRIC_BOUNDS[key][dataset])
        else:
            _min = min(self.METRIC_BOUNDS[key])
            _max = max(self.METRIC_BOUNDS[key])
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
        return {}

    def _query_benchmark(self, configuration: Union[ConfigSpace.Configuration, Dict],
                         fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                         rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Tuple[Dict, float]:

        # Query the benchmark
        result = self.jahs_benchmark(configuration, nepochs=fidelity['nepochs'], full_trajectory=False)

        # The last epoch contains the interesting results for us
        if self.kind == 'table':
            _id = list(result.keys())
            result_last_epoch = result[_id[0]]
        else:
            result_last_epoch = result[fidelity['nepochs']]
        _result_last_epoch = {}

        # Replace all accuracys with the misclassification rate (100 - acc)
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
        function_value = {
            key: self.normalize_metric(data=value, dataset=self.task, key=key)
            for key, value in function_value.items()
        }

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
        function_value = {
            key: self.normalize_metric(data=value, dataset=self.task, key=key)
            for key, value in function_value.items()
        }

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


class _JAHSMOSurrogateBenchmark(_SurrogateSearchSpace, _JAHSMOBenchmark):
    def __init__(self, **kwargs):
        super(_JAHSMOSurrogateBenchmark, self).__init__(
            kind='surrogate', metrics=['valid-acc', 'runtime', 'latency'], **kwargs,
        )


class _JAHSMOTabularBenchmark(_TabularSearchSpace, _JAHSMOBenchmark):
    def __init__(self, **kwargs):
        super(_JAHSMOTabularBenchmark, self).__init__(
            kind='table', metrics=['valid-acc', 'runtime', 'latency'], **kwargs,
        )


class JAHSMOCifar10SurrogateBenchmark(_JAHSMOSurrogateBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOCifar10SurrogateBenchmark, self).__init__(task='cifar10', rng=rng, **kwargs)


class JAHSMOColorectalHistologySurrogateBenchmark(_JAHSMOSurrogateBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOColorectalHistologySurrogateBenchmark, self).__init__(task='colorectal_histology', rng=rng, **kwargs)


class JAHSMOFashionMNISTSurrogateBenchmark(_JAHSMOSurrogateBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOFashionMNISTSurrogateBenchmark, self).__init__(task='fashion_mnist', rng=rng, **kwargs)


class JAHSMOCifar10TabularBenchmark(_JAHSMOTabularBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOCifar10TabularBenchmark, self).__init__(task='cifar10', rng=rng, **kwargs)


class JAHSMOColorectalHistologyTabularBenchmark(_JAHSMOTabularBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOColorectalHistologyTabularBenchmark, self).__init__(task='colorectal_histology', rng=rng, **kwargs)


class JAHSMOFashionMNISTTabularBenchmark(_JAHSMOTabularBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
        super(JAHSMOFashionMNISTTabularBenchmark, self).__init__(task='fashion_mnist', rng=rng, **kwargs)


__all__ = [
    "JAHSMOCifar10SurrogateBenchmark",
    "JAHSMOColorectalHistologySurrogateBenchmark",
    "JAHSMOFashionMNISTSurrogateBenchmark",
    "JAHSMOCifar10TabularBenchmark",
    "JAHSMOColorectalHistologyTabularBenchmark",
    "JAHSMOFashionMNISTTabularBenchmark",
]
