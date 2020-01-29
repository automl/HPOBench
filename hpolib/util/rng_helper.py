""" Helper functions to easily obtain randomState """
from typing import Union
import numpy as np


def get_rng(rng: Union[int, np.random.RandomState, None] = None,
            self_rng: Union[int, np.random.RandomState, None] = None) -> np.random.RandomState:
    """
    Helper function to obtain RandomState from int or create a new one.

    Sometimes a default random state (self_rng) is already available, but a new random state is desired.
    In this case rng is not None and not already a random state (int or None) -> a new random state is created.
    If rng is already a randomState, it is just returned.
    Same if rng is None, but the default rng is given.

    Parameters
    ----------
    rng: int, np.random.RandomState, None
    self_rng: np.random.RandomState, None

    Returns
    -------
    np.random.RandomState
    """

    if rng is not None:
        return _cast_int_to_random_state(rng)
    elif rng is None and self_rng is not None:
        return _cast_int_to_random_state(self_rng)
    else:
        return np.random.RandomState()


def _cast_int_to_random_state(rng: Union[int, np.random.RandomState]) -> np.random.RandomState:
    """
    Helper function to cast 'rng' from int to np.random.RandomState if necessary.

    Parameters
    ----------
    rng: int, np.random.RandomState

    Returns
    -------
    np.random.RandomState
    """
    if type(rng) == np.random.RandomState:
        return rng
    elif int(rng) == rng:
        # As seed is sometimes -1 (e.g. if SMAC optimizes a deterministic function) -> use abs()
        return np.random.RandomState(np.abs(rng))
    else:
        raise ValueError(f"{rng} is neither a number nor a RandomState. Initializing RandomState failed")
