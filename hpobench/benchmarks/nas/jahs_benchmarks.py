"""

Installation:
=============
``` pip install git+https://github.com/automl/jahs_bench_201.git ```
or along with hpobench by calling
`pip install -e .[jahs_bench_201] `

Version:
========
0.0.1:
    Initial commit using JAHS-Bench-201 version 1.0.2

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
        with requests.get(url, stream=True) as response:
            with open(save_tar_file, 'wb') as f:
                f.write(response.raw.read())

        logger.info("Download finished, extracting now")
        with tarfile.open(save_tar_file, 'r') as f:
            f.extractall(path=save_dir)

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

        result_dict = {
            'function_value': result_last_epoch,
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

        result_dict = {
            'function_value': result_last_epoch['valid-misclassification_rate'],
            'cost': cost,
            'info': {result_last_epoch}
        }

        return result_dict

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[ConfigSpace.Configuration, Dict],
                                fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity, rng, **kwargs)


class _CompleteSearchSpace:
    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        from jahs_bench.lib.core.configspace import joint_config_space
        joint_config_space.seed(seed)
        return joint_config_space


class _JAHSMOSurrogateBenchmark(_CompleteSearchSpace, _JAHSMOBenchmark):
    def __init__(self, task: str, **kwargs):
        super(_JAHSMOSurrogateBenchmark, self).__init__(
            task=task, kind='surrogate', metrics=['valid-acc', 'runtime', 'latency']
        )
        self.subset_metrics = ['valid-misclassification_rate', 'latency']

    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        result_dict = super(_JAHSMOSurrogateBenchmark, self).objective_function(
            configuration=configuration, fidelity=fidelity, rng=rng, **kwargs
        )
        result_dict['function_value'] = {
            key: value for key, value in result_dict['function_value'].items() if key in self.subset_metrics
        }
        result_dict['function_value'] = {
            key: self.normalize_metric(data=value, dataset=self.task, key=key)
            for key, value in result_dict['function_value'].items()
        }

        return result_dict


class JAHSMOCifar10SurrogateBenchmark(_JAHSMOSurrogateBenchmark):
    def __init__(self):
        super(JAHSMOCifar10SurrogateBenchmark, self).__init__(task='cifar10')


class JAHSMOColorectalHistologySurrogateBenchmark(_JAHSMOSurrogateBenchmark):
    def __init__(self):
        super(JAHSMOColorectalHistologySurrogateBenchmark, self).__init__(task='colorectal_histology')


class JAHSMOFashionMNISTSurrogateBenchmark(_JAHSMOSurrogateBenchmark):
    def __init__(self):
        super(JAHSMOFashionMNISTSurrogateBenchmark, self).__init__(task='fashion_mnist')


__all__ = [
    "JAHSMOCifar10SurrogateBenchmark",
    "JAHSMOColorectalHistologySurrogateBenchmark",
    "JAHSMOFashionMNISTSurrogateBenchmark",
]
