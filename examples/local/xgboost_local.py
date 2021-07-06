"""
Example with XGBoost (local)
============================
This example executes the xgboost benchmark locally with random configurations on the CC18 openml tasks.

To run this example please install the necessary dependencies via:
``pip3 install .[xgboost_example]``
"""

import argparse
from time import time

from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark as Benchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids


def run_experiment(on_travis: bool = False):
    task_ids = get_openmlcc18_taskids()
    for task_no, task_id in enumerate(task_ids):

        if on_travis and task_no == 5:
            break

        print(f'# ################### TASK {task_no + 1} of {len(task_ids)}: Task-Id: {task_id} ################### #')
        if task_id == 167204:
            continue  # due to memory limits

        b = Benchmark(task_id=task_id)
        cs = b.get_configuration_space()
        start = time()
        num_configs = 1
        for i in range(num_configs):
            configuration = cs.sample_configuration()
            print(configuration)
            for n_estimator in [8, 64]:
                for subsample in [0.4, 1]:
                    fidelity = {'n_estimators': n_estimator, 'dataset_fraction': subsample}
                    result_dict = b.objective_function(configuration.get_dictionary(),
                                                       fidelity=fidelity)
                    valid_loss = result_dict['function_value']
                    train_loss = result_dict['info']['train_loss']
                    assert result_dict['info']['fidelity'] == fidelity

                    result_dict = b.objective_function_test(configuration)
                    test_loss = result_dict['function_value']

                    print(f'[{i+1}|{num_configs}] No Estimator: {n_estimator:3d} - '
                          f'Subsample Rate: {subsample:.1f} - Test {test_loss:.4f} '
                          f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')
        print(f'Done, took totally {time()-start:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HPOBench CC Datasets', description='HPOBench on the CC18 data sets.',
                                     usage='%(prog)s --array_id <task_id>')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \'travis\'. This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)
