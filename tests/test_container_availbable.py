import subprocess
from hpolib import config_file
import logging


library = config_file.container_source


def search_container(container_name):
    out = subprocess.getoutput(f'singularity search {container_name}')
    logging.debug(out)

    out = out.split('\n\n')
    container_available = any((f'{library}/{container_name}' in line for line in out))
    return container_available


def test_availability():
    container_names = ['pybnn',
                       'svm_benchmark',
                       'xgboost_benchmark',
                       'nasbench_101',
                       'nasbench_201',
                       'tabular_benchmarks',
                       'cartpole',
                       'learna_benchmark'
                       ]

    all_available = True
    for container in container_names:
        container_available = search_container(container)

        if not container_available:
            logging.warning(f'Container for {container} is not found in {library}')
            all_available = False

    assert all_available, 'Some containers are not online available'
