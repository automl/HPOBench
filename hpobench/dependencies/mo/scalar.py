import numpy as np
from typing import Union

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except ImportError:
    print("scikit-learn not installed")


def get_fitted_scaler(x_train: np.ndarray, name: Union[None, str] = None):
    """
    Instantiates a scaler by a given name and fits the scaler with x_train.
    Parameters
    ----------
    x_train: np.ndarray
        Train data

    name: str, None
        Name of the scaling method. Defaults to no scaling.

    Returns
    -------

    """

    if name == "MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    elif name == "Standard":
        scaler = StandardScaler(copy=True)
    elif name is None or name == "None":
        return None
    else:
        raise NotImplementedError()

    scaler.fit(x_train)
    return lambda x: scaler.transform(x)
