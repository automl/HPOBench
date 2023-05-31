"""
Multiple Optimizers on SVMSurrogate
=======================================

This example shows how to run SMAC-HB and SMAC-random-search on SVMSurrogate

Please install the necessary dependencies via ``pip install .`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""
import logging
from pathlib import Path
from time import time

import numpy as np
from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.roar_facade import ROAR
from smac.intensification.hyperband import Hyperband
from smac.scenario.scenario import Scenario
from smac.callbacks import IncorporateRunResultCallback

from hpobench.container.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.example_utils import set_env_variables_to_use_only_one_core

logger = logging.getLogger("minicomp")
logging.basicConfig(level=logging.INFO)
set_env_variables_to_use_only_one_core()


class Callback(IncorporateRunResultCallback):
    def __init__(self):
        self.budget = 10

    def __call__(self, smbo, run_info, result, time_left) -> None:
        self.budget -= run_info.budget
        if self.budget < 0:
            # No budget left
            raise ValueError

def create_smac_rs(benchmark, output_dir: Path, seed: int):
    # Set up SMAC-HB
    cs = benchmark.get_configuration_space(seed=seed)

    scenario_dict = {"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 60,
                     "cs": cs,
                     "deterministic": "true",
                     "runcount-limit": 200,
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     "cutoff": 1800,  # runtime limit for target algorithm
                     "memory_limit": 10000,  # adapt this to reasonable value for your hardware
                     "output_dir": output_dir,
                     "abort_on_first_run_crash": True,
                     }

    scenario = Scenario(scenario_dict)
    def optimization_function_wrapper(cfg, seed, **kwargs):
        """ Helper-function: simple wrapper to use the benchmark with smac """
        result_dict = benchmark.objective_function(cfg, rng=seed)
        cs.sample_configuration()
        return result_dict['function_value']

    smac = ROAR(scenario=scenario,
                   rng=np.random.RandomState(seed),
                   tae_runner=optimization_function_wrapper,
                   )
    return smac

def create_smac_hb(benchmark, output_dir: Path, seed: int):
    # Set up SMAC-HB
    cs = benchmark.get_configuration_space(seed=seed)

    scenario_dict = {"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 60,
                     "cs": cs,
                     "deterministic": "true",
                     "runcount-limit": 200,
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     "cutoff": 1800,  # runtime limit for target algorithm
                     "memory_limit": 10000,  # adapt this to reasonable value for your hardware
                     "output_dir": output_dir,
                     "abort_on_first_run_crash": True,
                     }

    scenario = Scenario(scenario_dict)
    def optimization_function_wrapper(cfg, seed, instance, budget):
        """ Helper-function: simple wrapper to use the benchmark with smac """
        result_dict = benchmark.objective_function(cfg, rng=seed,
                                                   fidelity={"dataset_fraction": budget})
        cs.sample_configuration()
        return result_dict['function_value']

    smac = SMAC4MF(scenario=scenario,
                   rng=np.random.RandomState(seed),
                   tae_runner=optimization_function_wrapper,
                   intensifier=Hyperband,
                   intensifier_kwargs={'initial_budget': 0.1, 'max_budget': 1, 'eta': 3}
                   )
    return smac


def run_experiment(out_path: str, on_travis: bool = False):

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    hb_res = []
    rs_res = []
    for i in range(4):
        benchmark = SurrogateSVMBenchmark(rng=i)
        smac = create_smac_hb(benchmark=benchmark, seed=i, output_dir=out_path)
        callback = Callback()
        smac.register_callback(callback)
        try:
            smac.optimize()
        except ValueError:
            print("Done")
        incumbent = smac.solver.incumbent
        inc_res = benchmark.objective_function(configuration=incumbent)
        hb_res.append(inc_res["function_value"])

        benchmark = SurrogateSVMBenchmark(rng=i)
        smac = create_smac_rs(benchmark=benchmark, seed=i, output_dir=out_path)
        callback = Callback()
        smac.register_callback(callback)
        try:
            smac.optimize()
        except ValueError:
            print("Done")
        incumbent = smac.solver.incumbent
        inc_res = benchmark.objective_function(configuration=incumbent)
        rs_res.append(inc_res["function_value"])

    print("SMAC-HB", hb_res, np.median(hb_res))
    print("SMAC-RS", rs_res, np.median(rs_res))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='HPOBench - SVM comp',
                                     description='Run different opts on SVM Surrogate',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./svm_comp', type=str)
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, on_travis=args.on_travis)
