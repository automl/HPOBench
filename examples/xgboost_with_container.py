"""
Example with XGBoost (container)
================================

In this example, we show how to use a benchmark with a container. We provide container for some benchmarks.
They are hosted on https://cloud.sylabs.io/library/keggensperger/automl.

Furthermore, we use different fidelities to train the xgboost model - the number of estimators as well as the fraction
of training data points.

To use the container-example, you have to have singulartiy (>3.5) installed. Follow the official installation guide on
https://sylabs.io/guides/3.1/user-guide/quick_start.html#quick-installation-steps

Furthermore, make sure to install the right dependencies for the hpolib via:
``pip3 install .[xgboost_example,singularity]``.
"""


import logging
import time
import numpy as np
import json_tricks

import argparse
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

from hpolib.util.openml_data_manager import get_openmlcc18_taskids
from hpolib.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark


def run_benchmark(task_id: int, on_travis: bool = False):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    result_path = Path('./results')
    result_path.mkdir(exist_ok=True, parents=True)

    my_rng = np.random.RandomState(0)

    # It is possible to link to a custom container directory, by providing either a local path or link to another
    # hosting platform. Also, the name of the container can be specified.
    # The default value for  ``container source``  is defined in config.py and the ``container_name`` is defined in the
    # corresponding benchmark definition. Here, we use the default values to show how customize them.
    b = Benchmark(rng=my_rng,
                  container_source='library://keggensperger/automl/',
                  container_name='xgboost_benchmark',
                  task_id=task_id)

    start = time.time()
    cs = b.get_configuration_space()
    configuration = cs.get_default_configuration()

    n_estimators = [8, 64] if not on_travis else [8]
    subsample_ratios = [0.4, 1] if not on_travis else [0.4]

    result_per_data_set = []
    num_configs = 10 if not on_travis else 2
    for i in range(num_configs):
        data_per_config = {estimator: {subsample: {} for subsample in subsample_ratios} for estimator in n_estimators}
        for estimator in n_estimators:
            for subsample in subsample_ratios:
                try:
                    result_dict = b.objective_function(configuration, n_estimators=estimator, subsample=subsample)
                    valid_loss = result_dict['function_value']
                    train_loss = result_dict['train_loss']
                    result_dict = b.objective_function_test(configuration, n_estimators=estimator)
                    test_loss = result_dict['function_value']
                except:
                    train_loss, valid_loss, test_loss = -1024, -1024, -1024

                logger.info(f'[{i+1}|{num_configs}] No Estimator: {estimator:3d} - Subsample Rate: {subsample:.1f} '
                            f'- Test {test_loss:.4f} - Valid {valid_loss:.4f} - Train {train_loss:.4f}')

                result = {'train_loss': train_loss, 'valid_loss': valid_loss, 'test_loss': test_loss}
                data_per_config[estimator][subsample] = result

        result_per_data_set.append([configuration.get_dictionary(), data_per_config])

    data_result_path = result_path / f'{task_id}.json'
    with data_result_path.open('w') as fh:
        json_tricks.dump(result_per_data_set, fh)

    logger.info("Done, took totally %.2f s" % (time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HPOlib CC Datasets', description='HPOlib3 on the CC18 data sets.',
                                     usage='%(prog)s --array_id <task_id>')

    parser.add_argument('--array_id', default=0, type=int, help='Defines which data set to use. Values from 0 to 71')
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    task_ids = get_openmlcc18_taskids()
    if args.array_id < len(task_ids):
        run_benchmark(task_ids[args.array_id], on_travis=args.on_travis)
