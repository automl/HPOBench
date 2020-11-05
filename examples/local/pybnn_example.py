"""
Example with XGBoost (local)
============================
This example executes the xgboost benchmark locally with random configurations on the CC18 openml tasks.

To run this example please install the necessary dependencies via:
``pip3 install .[xgboost_example]``
"""

import argparse
from time import time

from hpobench.container.benchmarks.ml.pybnn import BNNOnToyFunction as Benchmark


def run_experiment(on_travis: bool = False):
    b = Benchmark(container_source='/media/philipp/Volume/Code/Container',
                  container_name='pybnn')
    cs = b.get_configuration_space()
    start = time()
    num_configs = 1

    configuration = cs.sample_configuration()
    fidelity = {'budget': 1000}
    result_dict = b.objective_function(configuration.get_dictionary(),
                                       fidelity=fidelity)
    print(result_dict)

    result_dict = b.objective_function_test(configuration)
    print(result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HPOBench CC Datasets', description='HPOBench on the CC18 data sets.',
                                     usage='%(prog)s --array_id <task_id>')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \'travis\'. This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)
