""" Helper functions to easily obtain randomState """
from typing import Union
import numpy as np


def get_rng(rng: Union[int, np.random.RandomState, None] = None,
            self_rng: Union[int, np.random.RandomState, None] = None) -> np.random.RandomState:
    """
    Helper function to obtain RandomState.
    Returns RandomState created from 'rng'
    If 'rng' then return 'RandomState' created from rng
    if rng is None returns self_rng
    if self_rng and rng is None return random RandomState

    Parameters
    ----------
    rng: int, np.random.RandomState, None
    self_rng: np.random.RandomState, None

    Returns
    -------
    np.random.RandomState
    """

    if rng is not None:
        return create_rng(rng)
    elif rng is None and self_rng is not None:
        return create_rng(self_rng)
    else:
        return np.random.RandomState()


def create_rng(rng: Union[int, np.random.RandomState, None]) -> np.random.RandomState:
    """
    Helper function to crate 'rng' from np.random.RandomState or int

    Parameters
    ----------
    rng: int, np.random.RandomState, None

    Returns
    -------
    np.random.RandomState
    """
    if rng is None:
        return np.random.RandomState()
    elif type(rng) == np.random.RandomState:
        return rng
    elif int(rng) == rng:
        # As seed is sometimes -1 (e.g. if SMAC optimizes a deterministic function
        rng = np.abs(rng)
        return np.random.RandomState(rng)
    else:
        raise ValueError(f"{rng} is neither a number nor a RandomState. Initializing RandomState failed")
