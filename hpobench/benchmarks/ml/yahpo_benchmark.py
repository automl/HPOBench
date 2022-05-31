"""
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

Prerequisites: 1) Install Conda
===============================
Conda environment in which the HPOBench is installed (pip install .). Activate your environment.
```
conda activate <Name_of_Conda_HPOBench_environment>
```

Prerequisites: 2) Install R
===========================

Install R (4.0.5 - IMPORTANT!) and the required dependencies:

``` bash
Rscript -e 'install.packages("remotes", repos = "http://cran.r-project.org")'

# Install OpenML dependencies
Rscript -e 'install.packages("curl", repos = "http://cran.r-project.org")' \
&& Rscript -e 'install.packages("httr", repos = "http://cran.r-project.org")' \
&& Rscript -e 'install.packages("farff", repos = "http://cran.r-project.org")' \
&& Rscript -e 'install.packages("OpenML", repos = "http://cran.r-project.org")' \

# Install rbv2 dependencies
Rscript -e 'remotes::install_version("BBmisc", version = "1.11", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("glmnet", version = "2.0-16", upgrade = "never", repos = "http://cran.r-project.o")' \
&& Rscript -e 'remotes::install_version("rpart", version = "4.1-13", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("e1071", version = "1.7-0.1", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("xgboost", version = "0.82.1", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("ranger", version = "0.11.2", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("RcppHNSW", version = "0.1.0", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("mlr", version = "2.14", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_github("mlr-org/mlr3misc", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_version("mlrCPO", version = "0.3.6", upgrade = "never", repos = "http://cran.r-projt.org")' \
&& Rscript -e 'remotes::install_github("pfistfl/rbv2", upgrade = "never")' \
&& Rscript -e 'remotes::install_version("testthat", version = "3.1.4", upgrade = "never", repos = "http://cran.r-project.org")' \
&& Rscript -e 'remotes::install_github("sumny/iaml", upgrade = "never")'
```
Prerequisites: 3) Install rpy2
==============================
Installing the connector between R and python might be a little bit tricky.
Official installation guide: https://rpy2.github.io/doc/latest/html/introduction.html

We received in some cases the error: "/opt/R/4.0.5/lib/R/library/methods/libs/methods.so: undefined symbol".
To solve this error, we had to execute the following command:
```
export LD_LIBRARY_PATH=$(python -m rpy2.situation LD_LIBRARY_PATH):${LD_LIBRARY_PATH}
```

1. Download data:
=================
Normally, the data will be downloaded automatically.

If you want to download the data on your own, you can download the data with the following command:

``` bash
git clone --depth 1 -b main https://github.com/pfistfl/yahpo_data.git
```

Later, you have to give yahpo the link to the data.

```python
from yahpo_gym import local_config
local_config.init_config()
local_config.set_data_path("path-to-data")
```

The data consist of surrogates for different data sets. Each surrogate is a compressed ONNX neural network.


2. Install HPOBench:
====================
```
git clone HPOBench
cd /path/to/HPOBench
pip install .[yahpo_gym_raw]
```

Changelog:
==========
0.0.1:
* First implementation
"""  # noqa: E501

import logging
from pathlib import Path
from typing import Union, Dict, List

import ConfigSpace as CS
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from yahpo_gym.benchmark_set import BenchmarkSet

import hpobench.config
from hpobench.abstract_benchmark import AbstractBenchmark, AbstractMultiObjectiveBenchmark

__version__ = '0.0.1'

logger = logging.getLogger('YAHPO-Raw')


class YAHPOGymMORawBenchmark(AbstractMultiObjectiveBenchmark):
    def __init__(self, scenario: str, instance: str,
                 rng: Union[np.random.RandomState, int, None] = None,
                 data_dir: Union[Path, str, None] = None):
        """
        Parameters
        ----------
        scenario : str
            Name for the learner. Must be one of [
            "rbv2_ranger", "rbv2_rpart", "rbv2_glmnet", "rbv2_xgboost", "rbv2_svm", "rbv2_aknn", "rbv2_super",
            "iaml_ranger", "iaml_rpart", "iaml_glmnet", "iaml_xgboost"
            ]
        instance : str
            A valid instance for the scenario. See `self.benchset.instances`.
            https://slds-lmu.github.io/yahpo_gym/scenarios.html#instances
        rng : np.random.RandomState, int, None
        """

        assert scenario.startswith('rbv2_') or scenario.startswith('iaml_'), \
            'Currently, we only support the experiments with rbv2_ and iaml from yahpo. ' \
            f'The scenario has to start with either rbv2_ or iaml_, but was {scenario}'

        from hpobench.util.data_manager import YAHPODataManager
        self.data_manager = YAHPODataManager(data_dir=data_dir)
        self.data_manager.load()

        self.scenario = scenario
        self.instance = instance
        self.benchset = BenchmarkSet(scenario, active_session=True)
        self.benchset.set_instance(instance)

        logger.info(f'Start Benchmark for scenario {scenario} and instance {instance}')
        super(YAHPOGymMORawBenchmark, self).__init__(rng=rng)

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

        # Cast python dict to R list:
        parameters = {**configuration, **fidelity}
        r_list = YAHPOGymMORawBenchmark.__cast_dict_to_rlist(parameters)

        # Call the random bot evaluation method
        if self.scenario.startswith('rbv2_'):
            # Establish a connection to the R package
            rbv2pkg = importr('rbv2')
            out = rbv2pkg.eval_yahpo(scenario=robjects.StrVector([self.scenario]), configuration=r_list)
        elif self.scenario.startswith('iaml_'):
            # We have to create a cache dir and initialize the cache
            _cache_dir = hpobench.config.config_file.cache_dir / 'R' / 'mlr3oml'
            oml = importr('mlr3oml')
            oml.initialize_cache(cache=robjects.StrVector([str(_cache_dir)]))

            iaml = importr('iaml')
            out = iaml.eval_yahpo(scenario=robjects.StrVector([self.scenario]), configuration=r_list)
        else:
            out = None

        # Cast the R list (result) back to a python dictionary
        result = YAHPOGymMORawBenchmark.__cast_to_dict(out)
        objectives = {target: value for target, value in result.items() if target in self.benchset.config.y_names}
        additional = {target: value for target, value in result.items() if target not in self.benchset.config.y_names}

        return {'function_value': objectives,
                "cost": result["timetrain"],
                'info': {'fidelity': fidelity, 'additional_info': additional}}

    @AbstractMultiObjectiveBenchmark.check_parameters
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
                         'https://github.com/pfistfl/rbv2/',
                         'https://github.com/sumny/iaml']
                }

    # pylint: disable=arguments-differ
    def get_objective_names(self) -> List[str]:
        return self.benchset.config.y_names

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
    def __cast_dict_to_rlist(py_dict):
        """ Convert a python dictionary to a RPy2 ListVector"""
        pairs = [f'{key} = {value}' if not isinstance(value, str) else f'{key} = \"{value}\"'
                 for key, value in py_dict.items()]
        pairs = ",".join(pairs)
        str_list = f"list({pairs})"
        r_list = robjects.r(str_list)
        return r_list


class YAHPOGymRawBenchmark(AbstractBenchmark):
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
            https://slds-lmu.github.io/yahpo_gym/scenarios.html#instances
        objective : str
            Name of the (single-crit) objective. See `self.benchset.config.y_names`.
            Initialized to None, picks the first element in y_names.
        rng : np.random.RandomState, int, None
        """
        self.backbone = YAHPOGymMORawBenchmark(scenario=scenario, instance=instance, rng=rng)
        self.objective = objective
        super(YAHPOGymRawBenchmark, self).__init__(rng=rng)

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
                'info': {'fidelity': fidelity,
                         'additional_info': mo_results['info']['additional_info'],
                         'objectives': mo_results['function_value']}}

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
        return YAHPOGymMORawBenchmark.get_meta_information()
