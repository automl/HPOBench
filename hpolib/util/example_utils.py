import os
from pathlib import Path
from typing import Dict


def get_travis_settings(type: str) -> Dict:
    """ Helper function to reduce time consumption for test runs on travis.ci"""
    if type == 'smac':
        return { "runcount-limit": 20, 'wallclock-limit': 100, 'cutoff': 100, 'memory_limit': 4000, 'output_dir': '.'}
    elif type == 'bohb':
        return {'max_budget': 3, 'num_iterations': 1, 'output_dir': Path('./')}
    else:
        raise ValueError(f'Unknown type {type}. Must be one of [smac, bohb]')


def set_env_variables_to_use_only_one_core():
    """ Helper function: Sets all variables which are responsible for using multiple threads to 1.
    This is necessary/useful, if you are computing on a cluster."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'