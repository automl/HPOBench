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
import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.benchmarks.rl.robot_pushing_utils import PushReward

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
        self.simulation = PushReward(gui=False)

        super(RobotPushingBenchmark, self).__init__(rng=rng)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        from hpobench.util.benchmarks.rl.robot_pushing_utils import PushReward
        xmin, xmax = PushReward.get_limits()

        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter('rx', lower=xmin[0], upper=xmax[0]),
            CS.UniformFloatHyperparameter('ry', lower=xmin[1], upper=xmax[1]),
            CS.UniformFloatHyperparameter('xvel', lower=xmin[2], upper=xmax[2]),
            CS.UniformFloatHyperparameter('yvel', lower=xmin[3], upper=xmax[3]),
            CS.UniformFloatHyperparameter('simu_steps', lower=xmin[4], upper=xmax[4]),
            CS.UniformFloatHyperparameter('init_angle', lower=xmin[5], upper=xmax[5]),
            CS.UniformFloatHyperparameter('rx2', lower=xmin[6], upper=xmax[6]),
            CS.UniformFloatHyperparameter('ry2', lower=xmin[7], upper=xmax[7]),
            CS.UniformFloatHyperparameter('xvel2', lower=xmin[8], upper=xmax[8]),
            CS.UniformFloatHyperparameter('yvel2', lower=xmin[9], upper=xmax[9]),
            CS.UniformFloatHyperparameter('simu_steps2', lower=xmin[10], upper=xmax[10]),
            CS.UniformFloatHyperparameter('init_angle2', lower=xmin[11], upper=xmax[11]),
            CS.UniformFloatHyperparameter('rtor', lower=xmin[12], upper=xmax[12]),
            CS.UniformFloatHyperparameter('rtor2', lower=xmin[13], upper=xmax[13]),
            ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        logger.debug('This benchmark has no fidelities.')
        return CS.ConfigurationSpace()

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'Robot Pushing Benchmark',
                'references': ['@inproceedings{wang2018batched,'
                               'title        = {Batched large-scale bayesian optimization in high-dimensional spaces},'
                               'author       = {Wang, Zi and Gehring, Clement and Kohli, Pushmeet '
                               '                and Jegelka, Stefanie},'
                               'booktitle    = {International Conference on Artificial Intelligence and Statistics},'
                               'pages        = {745--754},'
                               'year         = {2018},'
                               'organization = {PMLR}'
                               '}',

                               '@inproceedings{eriksson2019scalable,'
                               'title     = {Scalable Global Optimization via Local {Bayesian} Optimization},'
                               'author    = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D '
                               '             and Poloczek, Matthias},'
                               'booktitle = {Advances in Neural Information Processing Systems},'
                               'pages     = {5496--5507},'
                               'year      = {2019},'
                               'url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian'
                               '-optimization.pdf}'
                               '}'],
                'code': 'https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/master/test_functions'
                        'https://github.com/uber-research/TuRBO'
                }

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        from time import time

        start_time = time()

        distance_to_goal = self.simulation(configuration)

        finish_time = time()

        return {'function_value': distance_to_goal,
                "cost": finish_time - start_time,
                'info': {'fidelity': None}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        return self.objective_function(configuration, fidelity, rng, **kwargs)
