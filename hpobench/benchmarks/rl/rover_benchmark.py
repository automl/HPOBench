"""
This benchmark is from https://github.com/zi-w/Ensemble-Bayesian-Optimization/tree/master/test_functions
And is used in
- "Batched large-scale bayesian optimization in high-dimensional spaces", Wang et al.
- "Scalable Global Optimization via Local Bayesian Optimization", Eriksson et al.

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

Description of the benchmark:
=============================

From Batched large-scale bayesian optimization in high-dimensional spaces:
"... we consider a trajectory optimization task in 2D, meant to emulate a rover navigation task.
We describe a problem instance by defining a start position s and a goal position g as well as a cost function
over the state space. Trajectories are described by a set of points on which a BSpline is to be fitted.
By integrating the cost function over a given trajectory, we can compute the trajectory cost c(x) of a given trajectory
solution x ∈ [0, 1]^60."


1. Clone from github:
=====================
```
git clone git@github.com:automl/HPOBench.git
```

2. Clone and install
====================
```
cd /path/to/HPOBench
pip install .

```

Changelog:
==========
0.0.1:
* Initial implementation
"""

import logging
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.benchmarks.rl.rover_utils import get_obstacle_positions, AABoxes, NegGeom, UnionGeom, \
    ConstObstacleCost, ConstCost, AdditiveCosts, PointBSpline, RoverDomain, NormalizedInputFn, ConstantOffsetFn

__version__ = '0.0.1'

logger = logging.getLogger('RoverBenchmark')


class RoverBenchmark(AbstractBenchmark):

    N_COORDINATES = 30

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        rng : np.random.RandomState, int, None
        """
        super(RoverBenchmark, self).__init__(rng=rng)

        self.domain = self.create_large_domain(force_start=False,
                                               force_goal=False,
                                               start_miss_cost=self.l2cost,
                                               goal_miss_cost=self.l2cost,
                                               rng=self.rng)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        from time import time
        start_time = time()

        configuration = self.configuration_to_array(configuration)

        f = ConstantOffsetFn(self.domain, offset=5)
        f = NormalizedInputFn(f, self.domain.s_range)

        # This original problem aims to maximize the reward (negative costs). We cast it to a minimization task.
        # The optimal value is -5.
        reward = f(configuration)

        return {'function_value': -reward,
                "cost": time() - start_time,
                'info': {'fidelity': fidelity}}

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:

        return self.objective_function(configuration, fidelity=fidelity, rng=rng)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        A configuration in this benchmark is a set of 30 coordinates (60 points) defining the trajectory of the rover.
        A B-Spline is created from those coordinates, which represents then the route of the rover.

        We define here the 30 coordinates by 60 parameter. The coordinates are constructed from the values as follows:

        Values / Parameter:     coord_1_x, coord_1_y, coord_2_x, coord_2_y, ..., coord_30_y
        Resulting Coordinates:  (coord_1_x, coord_1_y), (coord_2_x, coord_2_y), ..., (coord_30_x, coord_30_y)

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([CS.UniformFloatHyperparameter(f'coord_{index}_{p}', lower=0, upper=1, log=False)
                                for index in range(1, RoverBenchmark.N_COORDINATES + 1) for p in ['x', 'y']])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace for the rover benchmark.

        Note that this benchmark has no fidelities! We return an empty configspace!

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)
        return fidel_space

    @staticmethod
    def get_meta_information():
        """ Returns the meta information for the benchmark """
        return {'name': 'Rover Benchmark',
                'references': ["@inproceedings{wang2018batched,  "
                               "title        = {Batched large-scale bayesian optimization in high-dimensional spaces},"
                               "author       = {Wang, Zi and Gehring, Clement and Kohli, Pushmeet and "
                               "                Jegelka, Stefanie},"
                               "booktitle    = {International Conference on Artificial Intelligence and Statistics},"
                               "pages        = {745--754},"
                               "year         = {2018},"
                               "organization = {PMLR}}",

                               "@inproceedings{eriksson2019scalable,"
                               "title     = {Scalable Global Optimization via Local {Bayesian} Optimization},"
                               "author    = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D "
                               "             and Poloczek, Matthias},"
                               "booktitle = {Advances in Neural Information Processing Systems},"
                               "pages     = {5496--5507},"
                               "year      = {2019},"
                               "url       = {http://papers.nips.cc/paper/"
                               "             8788-scalable-global-optimization-via-local-bayesian-optimization.pdf}}"
                               ],
                'code': ['https://github.com/zi-w/Ensemble-Bayesian-Optimization',
                         'https://github.com/uber-research/TuRBO']
                }

    @staticmethod
    def create_large_cost_function():
        # The center coordinates of some obstacles
        c = get_obstacle_positions()

        # Each obstacle is a box defined by two coordinates.
        # left lower corner: [center_x - 0.025, center_y - 0.025]
        # right upper corner: [center_x + 0.025, center_y + 0.025]
        low = c - 0.025
        high = c + 0.025

        # Create a map on which the rover should stay. lower left: [0, 0], upper right: [1, 1]
        r_box = np.array([[0.5, 0.5]])
        r_l = r_box - 0.5
        r_h = r_box + 0.5

        # Create the obstacles as Bounding Boxes
        trees = AABoxes(low, high)

        # If the Rover is not (!) in the world, then we have higher costs (that's why they use here "negative geometry")
        r_box = NegGeom(AABoxes(r_l, r_h))

        obstacles = UnionGeom([trees, r_box])

        # Define fix start and goal points
        start = np.array([0.05, 0.05])
        goal = np.array([0.95, 0.95])

        # The costs for hitting a obstacle is 20, having a longer trajectory (route) is 0.05
        costs = [ConstObstacleCost(obstacles, cost=20.), ConstCost(0.05)]
        cost_fn = AdditiveCosts(costs)
        return cost_fn, start, goal

    @staticmethod
    def create_large_domain(force_start=False,
                            force_goal=False,
                            start_miss_cost=None,
                            goal_miss_cost=None,
                            rng=None):

        cost_fn, start, goal = RoverBenchmark.create_large_cost_function()

        traj = PointBSpline(dim=2, num_points=RoverBenchmark.N_COORDINATES)

        domain = RoverDomain(cost_fn,
                             start=start,
                             goal=goal,
                             traj=traj,
                             start_miss_cost=start_miss_cost,
                             goal_miss_cost=goal_miss_cost,
                             force_start=force_start,
                             force_goal=force_goal,
                             s_range=np.array([[-0.1, -0.1], [1.1, 1.1]]),
                             rnd_stream=rng)

        return domain

    @staticmethod
    def configuration_to_array(configuration: Union[CS.Configuration, Dict]):
        if isinstance(configuration, CS.Configuration):
            return configuration.get_array()
        elif isinstance(configuration, Dict):
            configuration_array = np.zeros((RoverBenchmark.N_COORDINATES, 2))
            for coord_no in range(1, RoverBenchmark.N_COORDINATES):
                configuration_array[coord_no - 1, 0] = configuration[f'coord_{coord_no}_x']
                configuration_array[coord_no - 1, 1] = configuration[f'coord_{coord_no}_y']
            return configuration_array

    @staticmethod
    def l2cost(x, point):
        return 10 * np.linalg.norm(x - point, 1)
