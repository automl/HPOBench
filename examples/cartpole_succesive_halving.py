"""
SMAC on Cartpole with Succesive Halving
=======================================

This example shows the usage of an Hyperparameter Tuner, such as SMAC on the cartpole benchmark.
We use SMAC with Successive Halving.

Please install the necessary dependencies via ``pip install .[cartpole_example]``
"""
import logging
import numpy as np

from time import time
from pathlib import Path

from hpolib.util.example_utils import get_travis_settings, set_env_variables
from hpolib.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.hyperband import SuccessiveHalving
from smac.scenario.scenario import Scenario

logger = logging.getLogger("SH on cartpole")
logging.basicConfig(level=logging.INFO)
set_env_variables()


def run_experiment(out_path: str, on_travis: bool = False):

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    cs = Benchmark.get_configuration_space()

    scenario_dict = {"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 5*60*60,  # max duration to run the optimization (in seconds)
                     "cs": cs,  # configuration space
                     "deterministic": "true",
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     "cutoff": 1800,  # runtime limit for target algorithm
                     "memory_limit": 4000,  # adapt this to reasonable value for your hardware
                     "output_dir": str(out_path),
                     }

    if on_travis:
        scenario_dict.update(get_travis_settings('smac'))

    scenario = Scenario(scenario_dict)

    # Number of Agents, which are trained to solve the cartpole experiment
    max_budget = 5


    def optimization_function_wrapper(cfg, seed, instance, budget):
        """ Helper-function: simple wrapper to use the benchmark with smac"""
        b = Benchmark(rng=seed, max_budget=max_budget)
        result_dict = b.objective_function(cfg, budget=int(budget))
        return result_dict['function_value']


    smac = SMAC4HPO(scenario=scenario,
                    rng=np.random.RandomState(42),
                    tae_runner=optimization_function_wrapper,
                    intensifier=SuccessiveHalving,
                    intensifier_kwargs={'initial_budget': 1, 'max_budget': max_budget, 'eta': 3}
                    )

    start_time = time()
    # Example call of the function with default values. It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(), instance='1',  budget=1, seed=0)[1]
    print(f"Value for default configuration: {def_value:.4f}.\nEvaluation took {time() - start_time:.0f}s")

    # Start optimization
    start_time = time()
    try:
        smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    end_time = time()

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1', budget=max_budget, seed=0)[1]
    print(f"Value for optimized configuration: {inc_value:.4f}.\nOptimization took {end_time-start_time:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib - Successive Halving',
                                     description='HPOlib3 with SH on Cartpole',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./cartpole_smac_sh', type=str)
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, on_travis=args.on_travis)
