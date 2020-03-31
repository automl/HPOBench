import os
from pathlib import Path
from typing import Dict


def get_travis_settings(type: str) -> Dict:
    """ Helper function to reduce time consumption for test runs on travis.ci"""
    if type == 'smac':
        return {'wallclock-limit': 100, 'cutoff': 100, 'memory_limit': 4000, 'output_dir': '.'}
    elif type == 'bohb':
        return {'max_budget': 3, 'num_iterations': 1, 'output_dir': Path('./')}
    else:
        raise ValueError(f'Unknown type {type}. Must be one of [smac, bohb]')


def set_env_variables():
    """ Helper function: Sets all variables with are responsible for using multiple threads to 1.
    This is necessary/useful, if you are computing on a cluster."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
