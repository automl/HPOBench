""" OpenMLDataManager organizing the data for the benchmarks with data from OpenML-tasks.

DataManager organizing the download of the data.
The load function of a DataManger downloads the data given an unique OpenML identifier. It splits the data in
train, test and optional validation splits. It can be distinguished between holdout and cross-validation data sets.

For Non-OpenML data sets please use the hpolib.util.data_manager.
"""

from typing import Tuple, Union

import numpy as np
import openml
from sklearn.model_selection import train_test_split

import hpolib
from hpolib.util.data_manager import HoldoutDataManager, CrossvalidationDataManager
from hpolib.util.rng_helper import get_rng


def get_openml100_taskids():
    """
    Return taskids for the OpenML100 datasets
    See also here: https://www.openml.org/s/14
    Reference: https://arxiv.org/abs/1708.03731
    """
    return [
        258, 259, 261, 262, 266, 267, 271, 273, 275, 279, 283, 288, 2120,
        2121, 2125, 336, 75093, 75092, 75095, 75097, 75099, 75103, 75107,
        75106, 75109, 75108, 75112, 75129, 75128, 75135, 146574, 146575,
        146572, 146573, 146578, 146579, 146576, 146577, 75154, 146582,
        146583, 75156, 146580, 75159, 146581, 146586, 146587, 146584,
        146585, 146590, 146591, 146588, 146589, 75169, 146594, 146595,
        146592, 146593, 146598, 146599, 146596, 146597, 146602, 146603,
        146600, 146601, 75181, 146604, 146605, 75215, 75217, 75219, 75221,
        75225, 75227, 75231, 75230, 75232, 75235, 3043, 75236, 75239, 3047,
        232, 233, 236, 3053, 3054, 3055, 241, 242, 244, 245, 246, 248, 250,
        251, 252, 253, 254,
    ]


def get_openmlcc18_taskids():
    """
    Return taskids for the OpenML-CC18 datasets
    See also here: https://www.openml.org/s/99
    TODO: ADD reference
    """
    return [
        167149, 167150, 167151, 167152, 167153, 167154, 167155, 167156, 167157, 167158, 167159,
        167160, 167161, 167162, 167163, 167165, 167166, 167167, 167168, 167169, 167170, 167171,
        167164, 167173, 167172, 167174, 167175, 167176, 167177, 167178, 167179, 167180, 167181,
        167182, 126025, 167195, 167194, 167190, 167191, 167192, 167193, 167187, 167188, 126026,
        167189, 167185, 167186, 167183, 167184, 167196, 167198, 126029, 167197, 126030, 167199,
        126031, 167201, 167205, 189904, 167106, 167105, 189905, 189906, 189907, 189908, 189909,
        167083, 167203, 167204, 189910, 167202, 167097,
    ]


def _load_data(task_id):
    """ Helperfunction to load the data from the OpenML website. """
    task = openml.tasks.get_task(task_id)

    try:
        task.get_train_test_split_indices(fold=0, repeat=1)  # This should throw an ValueError!
        raise AssertionError(f'Task {task_id} has more than one repeat. This benchmark '
                             'can only work with a single repeat.')
    except ValueError:
        pass

    try:
        task.get_train_test_split_indices(fold=1, repeat=0)  # This should throw an ValueError!

        raise AssertionError(f'Task {task_id} has more than one fold. This benchmark '
                             'can only work with a single fold.')
    except ValueError:
        pass

    train_indices, test_indices = task.get_train_test_split_indices()

    X, y = task.get_X_and_y()

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # TODO replace by more efficient function which only reads in the data
    # saved in the arff file describing the attributes/features
    dataset = task.get_dataset()
    _, _, categorical_indicator, _ = dataset.get_data(target=task.target_name)

    variable_types = ['categorical' if ci else 'numerical' for ci in categorical_indicator]

    return X_train, y_train, X_test, y_test, variable_types, dataset.name


class OpenMLHoldoutDataManager(HoldoutDataManager):
    """ Base class for loading holdout data set from OpenML.

    Attributes
    ----------
    task_id : int
    rng : np.random.RandomState
    name : str
    variable_types : list
        Indicating the type of each feature in the loaded data (e.g. categorical, numerical)

    Parameters
    ----------
    openml_task_id : int
        Unique identifier for the task on OpenML
    rng : int, np.random.RandomState, None
        defines the random state
    """

    def __init__(self, openml_task_id: int, rng: Union[int, np.random.RandomState, None] = None):

        super(OpenMLHoldoutDataManager, self).__init__()

        self._save_to = hpolib.config_file.data_dir / 'OpenML'
        self.task_id = openml_task_id
        self.rng = get_rng(rng=rng)
        self.name = None
        self.variable_types = None

        self.create_save_directory(self._save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(str(self._save_to))

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads dataset from OpenML in config_file.data_directory.
        Downloads data if necessary.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """

        X_train, y_train, X_test, y_test, variable_types, name = _load_data(self.task_id)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.33,
                                                              random_state=self.rng)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_valid
        self.y_val = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.variable_types = variable_types
        self.name = name

        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test


class OpenMLCrossvalidationDataManager(CrossvalidationDataManager):
    """ Base class for loading cross-validation data set from OpenML.

    Attributes
    ----------
    task_id : int
    rng : np.random.RandomState
    name : str
    variable_types : list
        Indicating the type of each feature in the loaded data (e.g. categorical, numerical)

    Parameters
    ----------
    openml_task_id : int
        Unique identifier for the task on OpenML
    rng : int, np.random.RandomState, None
        defines the random state
    """

    def __init__(self, openml_task_id: int, rng: Union[int, np.random.RandomState, None] = None):
        super(OpenMLCrossvalidationDataManager, self).__init__()

        self._save_to = hpolib.config_file.data_dir / 'OpenML'
        self.task_id = openml_task_id
        self.rng = get_rng(rng=rng)
        self.name = None
        self.variable_types = None

        self.create_save_directory(self._save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(str(self._save_to))

    def load(self):
        """
        Loads dataset from OpenML in config_file.data_directory.
        Downloads data if necessary.
        """

        X_train, y_train, X_test, y_test, variable_types, name = _load_data(self.task_id)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.variable_types = variable_types
        self.name = name

        return self.X_train, self.y_train, self.X_test, self.y_test
