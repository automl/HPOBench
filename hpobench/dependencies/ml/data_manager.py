import openml
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from hpobench.util.data_manager import DataManager


from hpobench import config_file


class OpenMLDataManager(DataManager):

    def __init__(self, task_id: int,
                 valid_size: Union[float, None] = 0.33,
                 data_path: Union[str, Path, None] = None,
                 global_seed: Union[int, None] = 1):

        self.task_id = task_id
        self.global_seed = global_seed

        self.valid_size = valid_size

        self.train_X = None
        self.valid_X = None
        self.test_X = None
        self.train_y = None
        self.valid_y = None
        self.test_y = None
        self.train_idx = None
        self.test_idx = None
        self.task = None
        self.dataset = None
        self.preprocessor = None
        self.lower_bound_train_size = None
        self.n_classes = None

        if data_path is None:
            data_path = config_file.data_dir / "OpenML"

        self.data_path = data_path
        super(OpenMLDataManager, self).__init__()

    def load(self, valid_size=None, verbose=False):
        """Fetches data from OpenML and initializes the train-validation-test data splits

        The validation set is fixed till this function is called again or explicitly altered
        """
        # fetches task
        self.task = openml.tasks.get_task(self.task_id, download_data=False)
        self.n_classes = len(self.task.class_labels)

        # fetches dataset
        self.dataset = openml.datasets.get_dataset(self.task.dataset_id, download_data=False)
        if verbose:
            self.logger.debug(self.task)
            self.logger.debug(self.dataset)

        # check if the path to data splits is valid
        if self.data_path is not None and self.data_path.is_dir():
            data_path = self.data_path / str(self.task_id)
            required_file_list = [
                ("train", "x"), ("train", "y"),
                ("valid", "x"), ("valid", "y"),
                ("test", "x"), ("test", "y")
            ]
            for files in required_file_list:
                data_str = "{}_{}.parquet.gzip".format(*files)
                if (data_path / data_str).exists():
                    raise FileNotFoundError("{} not found!".format(data_str.format(*files)))
            # ignore the remaining data loaders and preprocessors as valid data splits available
            return

        # loads full data
        X, y, categorical_ind, feature_names = self.dataset.get_data(
            target=self.task.target_name, dataset_format="dataframe"
        )
        categorical_ind = np.array(categorical_ind)
        (cat_idx,) = np.where(categorical_ind)
        (cont_idx,) = np.where(~categorical_ind)

        # splitting dataset into train and test (10% test)
        # train-test split is fixed for a task and its associated dataset (from OpenML)
        self.train_idx, self.test_idx = self.task.get_train_test_split_indices()
        train_x = X.iloc[self.train_idx]
        train_y = y.iloc[self.train_idx]
        self.test_X = X.iloc[self.test_idx]
        self.test_y = y.iloc[self.test_idx]

        # splitting training into training and validation
        # validation set is fixed as per the global seed independent of the benchmark seed
        valid_size = self.valid_size if valid_size is None else valid_size
        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(
            train_x, train_y, test_size=valid_size, shuffle=True, stratify=train_y,
            random_state=check_random_state(self.global_seed)
        )

        # preprocessor to handle missing values, categorical columns encodings,
        # and scaling numeric columns
        self.preprocessor = make_pipeline(
            ColumnTransformer([
                (
                    "cat",
                    make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(sparse=False, handle_unknown="ignore")),
                    cat_idx.tolist(),
                ),
                (
                    "cont",
                    make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler()),
                    cont_idx.tolist(),
                )
            ])
        )
        if verbose:
            self.logger.debug("Shape of data pre-preprocessing: {}".format(self.train_X.shape))

        # preprocessor fit only on the training set
        self.train_X = self.preprocessor.fit_transform(self.train_X)
        # applying preprocessor built on the training set, across validation and test splits
        self.valid_X = self.preprocessor.transform(self.valid_X)
        self.test_X = self.preprocessor.transform(self.test_X)
        # converting boolean labels to strings
        self.train_y = self._convert_labels(self.train_y)
        self.valid_y = self._convert_labels(self.valid_y)
        self.test_y = self._convert_labels(self.test_y)

        # Similar to (https://arxiv.org/pdf/1605.07079.pdf)
        # use 10 times the number of classes as lower bound for the dataset fraction
        self.lower_bound_train_size = (10 * self.n_classes) / self.train_X.shape[0]
        self.lower_bound_train_size = np.max((1 / 512, self.lower_bound_train_size))

        if verbose:
            self.logger.debug("Shape of data post-preprocessing: {}".format(self.train_X.shape), "\n")
            self.logger.debug("\nTraining data (X, y): ({}, {})".format(self.train_X.shape, self.train_y.shape))
            self.logger.debug("Validation data (X, y): ({}, {})".format(self.valid_X.shape, self.valid_y.shape))
            self.logger.debug("Test data (X, y): ({}, {})".format(self.test_X.shape, self.test_y.shape))
            self.logger.debug("\nData loading complete!\n")
        return

    @staticmethod
    def _convert_labels(labels):
        """Converts boolean labels (if exists) to strings
        """
        label_types = list(map(lambda x: isinstance(x, bool), labels))
        if np.all(label_types):
            _labels = list(map(lambda x: str(x), labels))
            if isinstance(labels, pd.Series):
                labels = pd.Series(_labels, index=labels.index)
            elif isinstance(labels, np.array):
                labels = np.array(labels)
        return labels
