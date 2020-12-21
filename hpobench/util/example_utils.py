import os
from pathlib import Path
from typing import Dict


def get_travis_settings(optimizer_type: str) -> Dict:
    """ Helper function to reduce time consumption for test runs on travis.ci"""
    if optimizer_type == 'smac':
        return {"runcount-limit": 5, 'wallclock-limit': 50, 'cutoff': 50, 'memory_limit': 10000, 'output_dir': '.'}
    if optimizer_type == 'bohb':
        return {'max_budget': 2, 'num_iterations': 1, 'output_dir': Path('./')}

    raise ValueError(f'Unknown type {optimizer_type}. Must be one of [smac, bohb]')


def set_env_variables_to_use_only_one_core():
    """ Helper function: Sets all variables which are responsible for using multiple threads to 1.
    This is necessary/useful, if you are computing on a cluster."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
