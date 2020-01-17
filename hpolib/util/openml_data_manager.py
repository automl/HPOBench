""" OpenMLDataManager organizing the data for the benchmarks with data from OpenML-tasks.

DataManager organizing the download of the data.
The load function of a DataManger downloads the data given an unique OpenML identifier. It splits the data in
train, test and optional validation splits. It can be distinguished between holdout and cross-validation data sets.

For Non-OpenML data sets please use the hpolib.util.data_manager.
"""

from typing import Tuple

import numpy as np
import openml
from sklearn.model_selection import train_test_split

import hpolib
from hpolib.util.data_manager import HoldoutDataManager, CrossvalidationDataManager
from hpolib.util.rng_helper import get_rng


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

    def __init__(self, openml_task_id: int, rng: Tuple[int, np.random.RandomState, None] = None):

        super(OpenMLHoldoutDataManager, self).__init__()

        self._save_to = hpolib._config.data_dir / 'OpenML'
        self.task_id = openml_task_id
        self.rng = get_rng(rng=rng)
        self.name = None
        self.variable_types = None

        self.create_save_directory(self._save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(str(self._save_to))

    def load(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        """
        Loads dataset from OpenML in _config.data_directory.
        Downloads data if necessary.

        Returns
        -------
        X_train: np.array
        y_train: np.array
        X_val: np.array
        y_val: np.array
        X_test: np.array
        y_test: np.array
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


    def __init__(self, openml_task_id: int, rng: Tuple[int, np.random.RandomState, None] = None):
        super(OpenMLCrossvalidationDataManager, self).__init__()

        self._save_to = hpolib._config.data_dir / 'OpenML'
        self.task_id = openml_task_id
        self.rng = get_rng(rng=rng)
        self.name = None
        self.variable_types = None

        self.create_save_directory(self._save_to)

        openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'
        openml.config.set_cache_directory(str(self._save_to))

    def load(self):
        """
        Loads dataset from OpenML in _config.data_directory.
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
