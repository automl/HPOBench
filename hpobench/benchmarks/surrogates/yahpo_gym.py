"""
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

Prerequisites:
==============
Conda environment in which the HPOBench is installed (pip install .). Activate your environment.
```
conda activate <Name_of_Conda_HPOBench_environment>
```

1. Clone from github:
=====================
```
git clone HPOBench
```

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .[yahpo_gym]

```

Changelog:
==========
0.0.2:

* Add support for multi-objective benchmarks
* Add support for fairness benchmarks and interpretability benchmarks.
For these new benchmarks (fairness and interpretability), we recommend the following benchmarks and objectives:
For the entire list of available benchmarks, please take a look in the yahpo benchmark documentation.

Benchmark Name      |   Scenario    |   Objectives
--------------------|---------------|--------------
fair_fgrrm          | 7592          | mmce, feo
                    | 14965         | mmce, feo
--------------------|---------------|--------------
fair_rpart          | 317599        | mmce, ffomr
                    | 7592          | mmce, feo
--------------------|---------------|--------------
fair_ranger         | 317599        | mmce, fpredp
                    | 14965         | mmce, fpredp
--------------------|---------------|--------------
fair_xgboost        | 317599        | mmce, ffomr
                    | 7592          | mmce, ffnr
--------------------|---------------|--------------
fair_super          | 14965         | mmce, feo
                    | 317599        | mmce, ffnr
--------------------|---------------|--------------


Benchmark Name      |   Scenario    |   Objectives
--------------------|---------------|--------------
iaml_glmnet          | 1489         | mmce, nf
                    | 40981         | mmce, nf
--------------------|---------------|--------------
iaml_rpart          | 1489          | mmce, nf
                    | 41146         | mmce, nf
--------------------|---------------|--------------
iaml_ranger         | 40981         | mmce, nf
                    | 41146         | mmce, nf
--------------------|---------------|--------------
iaml_xgboost        | 40981         | mmce, nf
                    | 41146         | mmce, nf
--------------------|---------------|--------------
iaml_super          | 40981         | mmce, nf
                    | 41146         | mmce, nf
--------------------|---------------|--------------

0.0.1:
* First implementation
"""
import logging
from pathlib import Path
from typing import Union, Dict, List

import ConfigSpace as CS
import numpy as np
from yahpo_gym.benchmark_set import BenchmarkSet

from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark, AbstractSingleObjectiveBenchmark
from hpobench.util.data_manager import YAHPODataManager

__version__ = '0.0.2'

logger = logging.getLogger('YAHPOGym')


class YAHPOGymBaseBenchmark:
    def __init__(self, scenario: str, instance: str,
                 data_dir: Union[Path, str, None] = None,
                 multi_thread: bool = True,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Base Benchmark for all single and multi objective yahpo surrogate benchmarks.
        Parameters
        ----------
        scenario : str
            Name for the surrogate data. Must be one of
            ["lcbench", "fcnet", "nb301", "rbv2_svm",
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_aknn", "rbv2_xgboost", "rbv2_super",
            "fair_ranger", "fair_rpart", "fair_fgrrm",               "fair_xgboost", "fair_super",
            "iaml_ranger", "iaml_rpart", "iaml_glmnet",              "iaml_xgboost", "iaml_super"]
        instance : str
            A valid instance for the scenario. See `self.benchset.instances`.
        data_dir: Optional, str, Path
            Directory, where the yahpo data is stored.
            Download automatically from https://github.com/slds-lmu/yahpo_data/tree/fair
        multi_thread: bool
            Flag to run ONNX runtime with a single thread. Might be important on compute clusters.
            Defaults to True
        rng : np.random.RandomState, int, None
        """
        self.data_manager = YAHPODataManager(data_dir=data_dir)
        self.data_manager.load()

        self.scenario = scenario
        self.instance = instance
        self.benchset = BenchmarkSet(scenario, active_session=True, multithread=multi_thread)
        self.benchset.set_instance(instance)

        logger.info(f'Start Benchmark for scenario {scenario} and instance {instance}')
        super(YAHPOGymBaseBenchmark, self).__init__(rng=rng)

    # pylint: disable=arguments-differ
    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchset.get_opt_space(drop_fidelity_params=True, seed=seed)

    # pylint: disable=arguments-differ
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchset.get_fidelity_space(seed=seed)

    def _mo_objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        # No batch predicts, so we can grab the first item
        out = self.benchset.objective_function({**configuration, **fidelity})[0]
        # Convert to float for serialization
        out = {k: float(v) for k, v in out.items()}

        # Get runtime name
        cost = out[self.benchset.config.runtime_name]

        return {'function_value': out,
                "cost": cost,
                'info': {'fidelity': fidelity}}

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'YAHPO Gym',
                'references': ['@misc{pfisterer2021yahpo,',
                               'title={YAHPO Gym -- Design Criteria and a new Multifidelity '
                               '       Benchmark for Hyperparameter Optimization},',
                               'author    = {Florian Pfisterer and Lennart Schneider and'
                               '             Julia Moosbauer and Martin Binder'
                               '             and Bernd Bischl},',
                               'eprint={2109.03670},',
                               'archivePrefix={arXiv},',
                               'year      = {2021}}'],
                'code': 'https://github.com/pfistfl/yahpo_gym/yahpo_gym'}


class YAHPOGymMOBenchmark(YAHPOGymBaseBenchmark, AbstractMultiObjectiveBenchmark):

    def __init__(self, scenario: str, instance: str, objective: str = None,
                 data_dir: Union[Path, str, None] = None,
                 multi_thread: bool = True,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        For a list of available scenarios and instances see
        'https://slds-lmu.github.io/yahpo_gym/scenarios.html'
        Parameters
        ----------
        scenario : str
            Name for the surrogate data. Must be one of
            ["lcbench", "fcnet", "nb301", "rbv2_svm",
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_aknn", "rbv2_xgboost", "rbv2_super",
            "fair_ranger", "fair_rpart", "fair_fgrrm",               "fair_xgboost", "fair_super",
            "iaml_ranger", "iaml_rpart", "iaml_glmnet",              "iaml_xgboost", "iaml_super"]
        instance : str
            A valid instance for the scenario. See `self.benchset.instances`.
        data_dir: Optional, str, Path
            Directory, where the yahpo data is stored.
            Download automatically from https://github.com/slds-lmu/yahpo_data/tree/fair
        multi_thread: bool
            Flag to run ONNX runtime with a single thread. Might be important on compute clusters.
            Defaults to True
        rng : np.random.RandomState, int, None
        """
        self.objective = objective
        super(YAHPOGymMOBenchmark, self).__init__(scenario=scenario, instance=instance, rng=rng, data_dir=data_dir, multi_thread=multi_thread)

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        return self._mo_objective_function(configuration, fidelity, rng, **kwargs)

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) \
            -> Dict:
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)

    # pylint: disable=arguments-differ
    def get_objective_names(self) -> List[str]:
        return self.benchset.config.y_names


class YAHPOGymBenchmark(YAHPOGymBaseBenchmark, AbstractSingleObjectiveBenchmark):

    def __init__(self, scenario: str, instance: str, objective: str = None,
                 data_dir: Union[Path, str, None] = None,
                 multi_thread: bool = True,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        For a list of available scenarios and instances see
        'https://slds-lmu.github.io/yahpo_gym/scenarios.html'
        Parameters
        ----------
        scenario : str
            Name for the surrogate data. Must be one of
            ["lcbench", "fcnet", "nb301", "rbv2_svm",
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_aknn", "rbv2_xgboost", "rbv2_super",
            "fair_ranger", "fair_rpart", "fair_fgrrm",               "fair_xgboost", "fair_super",
            "iaml_ranger", "iaml_rpart", "iaml_glmnet",              "iaml_xgboost", "iaml_super"]
        instance : str
            A valid instance for the scenario. See `self.benchset.instances`.
        objective : str
            Name of the (single-crit) objective. See `self.benchset.config.y_names`.
            Initialized to None, picks the first element in y_names.
        data_dir: Optional, str, Path
            Directory, where the yahpo data is stored.
            Download automatically from https://github.com/slds-lmu/yahpo_data/tree/fair
        multi_thread: bool
            Flag to run ONNX runtime with a single thread. Might be important on compute clusters.
            Defaults to True
        rng : np.random.RandomState, int, None
        """
        self.objective = objective
        super(YAHPOGymBenchmark, self).__init__(scenario=scenario, instance=instance, rng=rng, data_dir=data_dir, multi_thread=multi_thread)

    @AbstractSingleObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        mo_results = self._mo_objective_function(configuration=configuration,
                                                 fidelity=fidelity,
                                                 **kwargs)

        # If not objective is set, we just grab the first returned entry.
        if self.objective is None:
            self.objective = self.benchset.config.y_names[0]

        obj_value = mo_results['function_value'][self.objective]

        return {'function_value': obj_value,
                "cost": mo_results['cost'],
                'info': {'fidelity': fidelity, 'objectives': mo_results['function_value']}}

    @AbstractSingleObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)
