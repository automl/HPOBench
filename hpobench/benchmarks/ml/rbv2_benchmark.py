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

1. Download data:
=================
The data will be downloaded automatically.

If you want to download the data on your own, you can download the data with the following command and then link the
hpobench-config's data-path to it.
You can download the requried data [here](https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/).

```python
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

The data consist of surrogates for different data sets. Each surrogate is a compressed ONNX neural network.


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

import logging
import os
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from yahpo_gym.benchmark_set import BenchmarkSet

from hpobench.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'

logger = logging.getLogger('YAHPO-Raw')


class YAHPOGymMORawBenchmark(AbstractBenchmark):

    def __init__(self, scenario: str, instance: str, objective: str = None,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
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

        # When in the containerized version, redirect to the data inside the container.
        if 'YAHPO_CONTAINER' in os.environ:
            from yahpo_gym.local_config import LocalConfiguration
            local_config = LocalConfiguration()
            local_config.init_config(data_path='/home/data/yahpo_data')

        self.scenario = scenario
        self.instance = instance
        self.benchset = BenchmarkSet(scenario, active_session=True)
        self.benchset.set_instance(instance)
        self.objective = objective
        logger.info(f'Start Benchmark for scenario {scenario} and instance {instance}')
        super(YAHPOGymMORawBenchmark, self).__init__(rng=rng)

    # pylint: disable=arguments-differ
    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchset.get_opt_space(drop_fidelity_params=True, seed=seed)

    # pylint: disable=arguments-differ
    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return self.benchset.get_fidelity_space(seed=seed)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        # Establish a connection to the R package
        rbv2pkg = importr('rbv2')

        # Cast python dict to R list:
        parameters = {**configuration, **fidelity}
        list_str = YAHPOGymMORawBenchmark.__cast_dict_to_rlist(parameters)

        # Call the random bot evaluation method
        out = rbv2pkg.eval_yahpo(scenario=robjects.StrVector([self.scenario]), configuration=list_str)

        # Cast the R list (result) back to a python dictionary
        result = YAHPOGymMORawBenchmark.__cast_to_dict(out)

        if self.objective is None:
            self.objective = self.benchset.config.y_names[0]
        obj_value = result[self.objective]

        return {'function_value': obj_value,
                "cost": result["timetrain"],
                'info': {'fidelity': fidelity, 'objectives': result}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        return self.objective_function(configuration, fidelity=fidelity, rng=rng)

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'YAHPO Gym',
                'references': ['@misc{pfisterer2021yahpo,',
                               'title={YAHPO Gym -- Design Criteria and a new Multifidelity Benchmark '
                               '       for Hyperparameter Optimization},',
                               'author={Florian Pfisterer and Lennart Schneider and Julia Moosbauer '
                               '        and Martin Binder and Bernd Bischl},',
                               'eprint={2109.03670},',
                               'archivePrefix={arXiv},',
                               'year={2021}}'],
                'code': ['https://github.com/pfistfl/yahpo_gym/yahpo_gym',
                         'https://github.com/pfistfl/rbv2/']
                }

    @staticmethod
    def __cast_to_dict(r_list_object):
        """
        Convert an RPy2 ListVector to a Python dict.
        Source: https://ogeek.cn/qa/?qa=815151/
        """
        result = {}
        for i, name in enumerate(r_list_object.names):
            if isinstance(r_list_object[i], robjects.ListVector):
                result[name] = YAHPOGymMORawBenchmark.__cast_to_dict(r_list_object[i])
            elif len(r_list_object[i]) == 1:
                result[name] = r_list_object[i][0]
            else:
                result[name] = r_list_object[i]
        return result

    @staticmethod
    def __cast_dict_to_rlist(py_dict) -> str:
        """ Convert a python dictionary to a RPy2 ListVector in str representation. """
        pairs = [f'{key} = {value}' if not isinstance(value, str) else f'{key} = \"{value}\"'
                 for key, value in py_dict.items()]
        pairs = ",".join(pairs)
        cmd = f"list({pairs})"
        return cmd
