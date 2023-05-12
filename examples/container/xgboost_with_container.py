"""
Example with XGBoost (container)
================================

In this example, we show how to use a benchmark with a container. We provide container for some benchmarks.
They are hosted on https://cloud.sylabs.io/library/muelleph/automl.

Furthermore, we use different fidelities to train the xgboost model - the number of estimators as well as the fraction
of training data points.

To use the container-example, you have to have singulartiy (>3.5) installed. Follow the official installation guide on
https://sylabs.io/guides/3.1/user-guide/quick_start.html#quick-installation-steps

Furthermore, make sure to install the right dependencies for the hpobench via:
``pip3 install .``.
"""

import argparse
import logging
from time import time

from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids

logging.basicConfig(level=logging.INFO)


def run_experiment(on_travis: bool = False):
    task_ids = get_openmlcc18_taskids()
    for task_no, task_id in enumerate(task_ids):

        if on_travis and task_no == 5:
            break

        print(f'# ################### TASK {task_no + 1} of {len(task_ids)}: Task-Id: {task_id} ################### #')
        if task_id == 167204:
            continue  # due to memory limits

        b = Benchmark(task_id=task_id,
                      container_name='xgboost_benchmark',
                      container_source='library://phmueller/automl')

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

                    result_dict = b.objective_function_test(configuration, fidelity={'n_estimators': n_estimator})
                    test_loss = result_dict['function_value']

                    print(f'[{i+1}|{num_configs}] No Estimator: {n_estimator:3d} - '
                          f'Subsample Rate: {subsample:.1f} - Test {test_loss:.4f} '
                          f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')
        b.__del__()
        print(f'Done, took totally {time()-start:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='HPOBench CC Datasets', description='HPOBench on the CC18 data sets.',
                                     usage='%(prog)s --array_id <task_id>')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \'travis\'. This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)
