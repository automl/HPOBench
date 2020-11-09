"""
SMAC on Cartpole with Hyperband
===============================

This example shows the usage of an Hyperparameter Tuner, such as SMAC on the cartpole benchmark.
We use SMAC with Hyperband.

Please install the necessary dependencies via ``pip install .`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""
import logging
from pathlib import Path
from time import time

import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.hyperband import Hyperband
from smac.scenario.scenario import Scenario

from hpobench.container.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from hpobench.util.example_utils import get_travis_settings, set_env_variables_to_use_only_one_core

logger = logging.getLogger("HB on cartpole")
logging.basicConfig(level=logging.INFO)
set_env_variables_to_use_only_one_core()


def run_experiment(out_path: str, on_travis: bool = False):

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    benchmark = Benchmark(container_source='library://phmueller/automl',
                          container_name='cartpole',
                          rng=1)

    cs = benchmark.get_configuration_space(seed=1)

    scenario_dict = {"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 5 * 60 * 60,  # max duration to run the optimization (in seconds)
                     "cs": cs,  # configuration space
                     "deterministic": "true",
                     "runcount-limit": 200,
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     "cutoff": 1800,  # runtime limit for target algorithm
                     "memory_limit": 10000,  # adapt this to reasonable value for your hardware
                     "output_dir": str(out_path),
                     }

    if on_travis:
        scenario_dict.update(get_travis_settings('smac'))

    scenario = Scenario(scenario_dict)

    # Number of Agents, which are trained to solve the cartpole experiment
    max_budget = 5 if not on_travis else 2

    def optimization_function_wrapper(cfg, seed, instance, budget):
        """ Helper-function: simple wrapper to use the benchmark with smac"""

        # Now that we have already downloaded the conatiner,
        # we only have to start a new instance. This is a fast operation.
        b = Benchmark(container_source='library://phmueller/automl',
                      container_name='cartpole',
                      rng=seed)

        # Old API ---- NO LONGER SUPPORTED ---- This will simply ignore the fidelities
        # result_dict = b.objective_function(cfg, budget=int(budget))

        # New API ---- Use this
        result_dict = b.objective_function(cfg, fidelity={"budget": int(budget)})
        return result_dict['function_value']

    smac = SMAC4HPO(scenario=scenario,
                    rng=np.random.RandomState(42),
                    tae_runner=optimization_function_wrapper,
                    intensifier=Hyperband,  # you can also change the intensifier to use like this!
                    intensifier_kwargs={'initial_budget': 1, 'max_budget': max_budget, 'eta': 3}
                    )

    start_time = time()
    try:
        smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    end_time = time()

    if not on_travis:
        inc_value = smac.get_tae_runner().run(config=incumbent, instance='1', budget=max_budget, seed=0)[1]
        print(f"Value for optimized configuration: {inc_value:.4f}.\n"
              f"Optimization took {end_time-start_time:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='HPOBench - Hyperband',
                                     description='HPOBench with HB on Cartpole',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./cartpole_smac_hb', type=str)
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, on_travis=args.on_travis)
