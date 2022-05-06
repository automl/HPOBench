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
0.0.1:
* First implementation
"""
import os
import logging
from typing import Union, Dict, List

import ConfigSpace as CS
import numpy as np

from yahpo_gym.benchmark_set import BenchmarkSet

from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark, AbstractBenchmark
__version__ = '0.0.1'

logger = logging.getLogger('YAHPOGym')


class YAHPOGymMOBenchmark(AbstractMultiObjectiveBenchmark):

    def __init__(self, scenario: str, instance: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        For a list of available scenarios and instances see
        'https://slds-lmu.github.io/yahpo_gym/scenarios.html'
        Parameters
        ----------
        scenario : str
            Name for the surrogate data. Must be one of ["lcbench", "fcnet", "nb301", "rbv2_svm",
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_aknn", "rbv2_xgboost", "rbv2_super"]
        instance : str
            A valid instance for the scenario. See `self.benchset.instances`.
        rng : np.random.RandomState, int, None
        """

        # When in the containerized version, redirect to the data inside the container.
        if 'YAHPO_CONTAINER' in os.environ:
            from yahpo_gym.local_config import LocalConfiguration
            local_config = LocalConfiguration()
            local_config.init_config(data_path='/home/data/yahpo_data')

        self.scenario = scenario
        self.instance = instance
        self.benchset = BenchmarkSet(scenario, active_session=True)
        self.benchset.set_instance(instance)

        logger.info(f'Start Benchmark for scenario {scenario} and instance {instance}')
        super(YAHPOGymMOBenchmark, self).__init__(rng=rng)

    # pylint: disable=arguments-differ
    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchset.get_opt_space(drop_fidelity_params=True, seed=seed)

    # pylint: disable=arguments-differ
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchset.get_fidelity_space(seed=seed)

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
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

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) \
            -> Dict:
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)

    # pylint: disable=arguments-differ
    def get_objective_names(self) -> List[str]:
        return self.benchset.config.y_names

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


class YAHPOGymBenchmark(AbstractBenchmark):

    def __init__(self, scenario: str, instance: str, objective: str = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        For a list of available scenarios and instances see
        'https://slds-lmu.github.io/yahpo_gym/scenarios.html'
        Parameters
        ----------
        scenario : str
            Name for the surrogate data. Must be one of ["lcbench", "fcnet", "nb301", "rbv2_svm",
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_aknn", "rbv2_xgboost", "rbv2_super"]
        instance : str
            A valid instance for the scenario. See `self.benchset.instances`.
        objective : str
            Name of the (single-crit) objective. See `self.benchset.config.y_names`.
            Initialized to None, picks the first element in y_names.
        rng : np.random.RandomState, int, None
        """

        self.backbone = YAHPOGymMOBenchmark(scenario=scenario, instance=instance, rng=rng)
        self.objective = objective

        super(YAHPOGymBenchmark, self).__init__(rng=rng)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        mo_results = self.backbone.objective_function(configuration=configuration,
                                                      fidelity=fidelity,
                                                      **kwargs)

        # If not objective is set, we just grab the first returned entry.
        if self.objective is None:
            self.objective = self.backbone.benchset.config.y_names[0]

        obj_value = mo_results['function_value'][self.objective]

        return {'function_value': obj_value,
                "cost": mo_results['cost'],
                'info': {'fidelity': fidelity, 'objectives': mo_results['function_value']}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)

    # pylint: disable=arguments-differ
    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.backbone.get_configuration_space(seed=seed)

    # pylint: disable=arguments-differ
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.backbone.get_fidelity_space(seed=seed)

    @staticmethod
    def get_meta_information() -> Dict:
        return YAHPOGymMOBenchmark.get_meta_information()
