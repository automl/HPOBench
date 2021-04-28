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
pip install pygame==2.0.1 box2d-py==2.3.8
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
pip install .[paramnet]

```

Changelog:
==========

0.0.1:
* First implementation
"""
import warnings
import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark

__version__ = '0.0.1'

logger = logging.getLogger('RobotPushing')


class RobotPushingBenchmark(AbstractBenchmark):

    def __init__(self,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        rng : np.random.RandomState, int, None
        """


        super(RobotPushingBenchmark, self).__init__(rng=rng)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'Robot Pushing Benchmark',
                'references': [],
                'code': ''
                        ''
                }

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        obj_value = 0
        cost = 0

        return {'function_value': obj_value,
                "cost": cost,
                'info': {'fidelity': fidelity}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        return self.objective_function(configuration, fidelity={'step': 50}, rng=rng)
