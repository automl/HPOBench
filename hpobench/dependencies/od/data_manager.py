import os
from typing import Union
from urllib.request import urlretrieve

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

import hpobench
from hpobench.util.data_manager import CrossvalidationDataManager
from hpobench.util.rng_helper import get_rng

ODDS_PATH = 'https://automl.org/wp-content/uploads/datasets/ODDS'
ODDS_NAMES = [
    "annthyroid", "arrhythmia",  "breastw", "cardio", "ionosphere",
    "mammography", "musk", "optdigits", "pendigits", "pima",
    "satellite", "satimage-2", "thyroid", "vowels", "wbc"]
ODDS_URL = {name: os.path.join(ODDS_PATH, name + ".mat") for name in ODDS_NAMES}


class OutlierDetectionDataManager(CrossvalidationDataManager):
    """ Base class for loading dataset from ODDS. """

    def __init__(self, dataset_name: str, rng: Union[int, np.random.RandomState, None] = None):
        """
        Parameters
        ----------
        dataset_name : str
            Must be one of [
                "annthyroid", "arrhythmia", "breastw", "cardio", "ionosphere",
                "mammography", "musk", "optdigits", "pendigits", "pima",
                "satellite", "satimage-2", "thyroid", "vowels", "wbc"]
        rng : int, np.random.RandomState, None
            defines the random state
        """

        super(OutlierDetectionDataManager, self).__init__()

        if dataset_name not in ODDS_URL:
            raise NotImplementedError()

        self.dataset_name = dataset_name
        self.rng = get_rng(rng=rng)

        self._url_source = ODDS_URL[dataset_name]
        self._save_to = hpobench.config_file.data_dir / dataset_name

        self.create_save_directory(self._save_to)
        filename = self.__load_data(filename=self.dataset_name)

        self.dataset = OutlierDataset(filename, rng=self.rng, logger=self.logger)

    def __load_data(self, filename: str, images: bool = False) -> np.ndarray:
        save_filename = self._save_to / filename
        if not save_filename.exists():
            self.logger.debug(f"Downloading {self._url_source + filename} "
                              f"to {save_filename}")
            urlretrieve(self._url_source, str(save_filename))
        else:
            self.logger.debug(f"Load data {save_filename}")

        return save_filename

    def load(self):
        raise NotImplementedError()


class OutlierDataset:
    def __init__(self, filename,
                 rng,
                 normal_classes=[0],
                 outlier_classes=[1],
                 contamination_ratio=0.05,
                 val_split_ratio=0.2,
                 test_split_ratio=0.2,
                 logger=None):

        """
        Prepares the data for outlier detection.
        Args:
            normal_classes: Existing labels will be changed to 0.
            outlier_classes: Existing labels will be changed to 1.
            contamination_ratio: The ratio of outliers in the training dataset. If set to `None`, ratio won't changed.
                If ratio is higher than data available, the highest possible ratio is used.
            split_ratio: Ratio how much test data of the given data will be used.
            seed: Defines the initial shuffle of the data.
        """

        assert len(normal_classes) > 0 and len(outlier_classes) > 0
        assert 0.0 < val_split_ratio < 1.0
        assert 0.0 < test_split_ratio < 1.0

        data = scipy.io.loadmat(filename)
        X, y = data["X"], data["y"]

        if contamination_ratio is not None:
            assert contamination_ratio < 1.0 and contamination_ratio >= 0.0

        self.rng = rng
        self.logger = logger
        self.contamination_ratio = contamination_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio

        # Initial preprocessing
        X = np.array(X).astype(float)
        y = np.array(y).astype(int)
        y = np.reshape(y, (y.shape[0], 1))

        self.logger.info("Clean data and transform to outlier task.")
        # Clean data first
        class_indices = [i in normal_classes + outlier_classes for i in y.flatten()]
        X = X[class_indices]
        y = y[class_indices]

        # Replace normal classes with 0
        normal_indices = [i in normal_classes for i in y.flatten()]
        y[normal_indices] = 0

        # Replace outlier classes with 1
        outlier_indices = [i in outlier_classes for i in y.flatten()]
        y[outlier_indices] = 1

        normal_indices = np.where(y == 0)[0]
        outlier_indices = np.where(y == 1)[0]
        num_normal = len(normal_indices)
        num_outlier = len(outlier_indices)
        # num_all = num_normal + num_outlier

        self.logger.info(f"Found {num_normal} normal data.")
        self.logger.info(f"Found {num_outlier} anomaly data.")

        self.logger.info("Begin data splitting.")
        test_normal_cut = int(len(normal_indices)*(1-test_split_ratio))
        test_outlier_cut = int(len(outlier_indices)*(1-test_split_ratio))
        test_normal_indices = normal_indices[test_normal_cut:]
        test_outlier_indices = outlier_indices[test_outlier_cut:]

        # Now separate train and test data
        train_normal_indices = normal_indices[:test_normal_cut]
        train_outlier_indices = outlier_indices[:test_outlier_cut]

        # Based on the seed, we shuffle the training dataset now
        self.logger.info("Shuffle data.")
        perm = rng.permutation(len(train_normal_indices))
        train_normal_indices = train_normal_indices[perm]

        perm = rng.permutation(len(train_outlier_indices))
        train_outlier_indices = train_outlier_indices[perm]

        # train_split_ratio = (1 - (val_split_ratio + test_split_ratio))
        # train_split_ratio_scaled = train_split_ratio / (train_split_ratio + val_split_ratio)

        self.train_normal_indices = train_normal_indices
        self.train_outlier_indices = train_outlier_indices
        self.test_normal_indices = test_normal_indices
        self.test_outlier_indices = test_outlier_indices

        self.logger.info(f"Found training data: {len(train_normal_indices)} normal "
                         f"and {len(train_outlier_indices)} anomaly data.")
        self.logger.info(f"Found test data: {len(test_normal_indices)} normal "
                         f"and {len(test_outlier_indices)} anomaly data.")

        self._X = X
        self._y = y

    def get_contamination_ratio(self):
        return self.contamination_ratio

    def get_features(self):
        return self._X.shape[1]

    def get_train_val_data(self, split=None, max_splits=4):

        if split is None or split == 0:
            assert self.val_split_ratio
            self.logger.info("Using default train val with split")

            # Calculate the ratio based on val and test split ratio
            train_split_ratio = (1 - (self.val_split_ratio + self.test_split_ratio))
            train_split_ratio_scaled = train_split_ratio / (train_split_ratio + self.val_split_ratio)
            val_split_ratio_scaled = 1 - train_split_ratio_scaled
            split = 1
        else:
            assert split > 0 and split <= max_splits
            self.logger.info(f"Using train val with split {split}")
            val_split_ratio_scaled = 1 / max_splits

        _val_normal_indices = np.arange(
            int(len(self.train_normal_indices)*(split-1)*val_split_ratio_scaled),
            int(len(self.train_normal_indices)*split*val_split_ratio_scaled)
        )

        _val_outlier_indices = np.arange(
            int(len(self.train_outlier_indices)*(split-1)*val_split_ratio_scaled),
            int(len(self.train_outlier_indices)*split*val_split_ratio_scaled),
        )

        _train_normal_indices = np.arange(0, len(self.train_normal_indices))
        _train_outlier_indices = np.arange(0, len(self.train_outlier_indices))

        # Now remove the val indices
        _train_normal_indices = np.delete(_train_normal_indices, _val_normal_indices)
        _train_outlier_indices = np.delete(_train_outlier_indices, _val_outlier_indices)

        # Now we can use the selecte indices to get our "real" indices
        # (which were shuffled before)
        val_normal_indices = self.train_normal_indices[_val_normal_indices].tolist()
        val_outlier_indices = self.train_outlier_indices[_val_outlier_indices].tolist()

        train_normal_indices = self.train_normal_indices[_train_normal_indices].tolist()
        train_outlier_indices = self.train_outlier_indices[_train_outlier_indices].tolist()
        num_train_normal = len(train_normal_indices)
        num_train_outlier = len(train_outlier_indices)

        contamination_ratio = self.contamination_ratio
        if contamination_ratio is not None:
            self.logger.debug(f"Prepare train dataset with {contamination_ratio*100}% outlier fraction.")

            if contamination_ratio == 0:
                train_outlier_indices = []
                num_train_outlier = 0
            else:
                # Equation based on: num_outlier / (num_outlier + num_normal) = x
                required_num_train_outlier = int(contamination_ratio * num_train_normal / (1-contamination_ratio))
                required_num_train_normal = int(num_train_outlier * (1-contamination_ratio) / contamination_ratio)
                if num_train_normal < required_num_train_normal:
                    # Cut the outlier
                    train_outlier_indices = train_outlier_indices[:required_num_train_outlier]
                    num_train_outlier = len(train_outlier_indices)
                    self.logger.debug(f"Changed anomaly training data to {num_train_outlier}.")
                else:
                    # Cut the normals
                    train_normal_indices = train_normal_indices[:required_num_train_normal]
                    num_train_normal = len(train_normal_indices)
                    self.logger.debug(f"Changed normal training data to {num_train_normal}.")
        else:
            num_normal = len(self.train_normal_indices) + len(self.test_normal_indices)
            num_outlier = len(self.train_outlier_indices) + len(self.test_outlier_indices)
            contamination_ratio = num_outlier / (num_normal + num_outlier)
            self.logger.debug(f"Dataset is used with standard outlier fraction {contamination_ratio*100}%.")

        self.logger.debug(f"Using training data: {len(train_normal_indices)} normal "
                          f"and {len(train_outlier_indices)} anomaly data.")
        self.logger.debug(f"Using validation data: {len(val_normal_indices)} normal "
                          f"and {len(val_outlier_indices)} anomaly data.")

        X_train = self._X[train_normal_indices+train_outlier_indices]
        y_train = self._y[train_normal_indices+train_outlier_indices].flatten()

        X_val = self._X[val_normal_indices+val_outlier_indices]
        y_val = self._y[val_normal_indices+val_outlier_indices].flatten()

        return (X_train, y_train), (X_val, y_val)

    def get_train_data(self):
        train_normal_indices = self.train_normal_indices.tolist()
        train_outlier_indices = self.train_outlier_indices.tolist()
        num_train_normal = len(train_normal_indices)
        num_train_outlier = len(train_outlier_indices)

        contamination_ratio = self.contamination_ratio
        if contamination_ratio is not None:
            self.logger.debug(f"Prepare train dataset with {contamination_ratio*100}% outlier fraction.")

            if contamination_ratio == 0:
                train_outlier_indices = []
                num_train_outlier = 0
            else:
                # Equation based on: num_outlier / (num_outlier + num_normal) = x
                required_num_train_outlier = int(contamination_ratio * num_train_normal / (1-contamination_ratio))
                required_num_train_normal = int(num_train_outlier * (1-contamination_ratio) / contamination_ratio)
                if num_train_normal < required_num_train_normal:
                    # Cut the outlier
                    train_outlier_indices = train_outlier_indices[:required_num_train_outlier]
                    num_train_outlier = len(train_outlier_indices)
                    self.logger.debug(f"Changed anomaly training data to {num_train_outlier}.")
                else:
                    # Cut the normals
                    train_normal_indices = train_normal_indices[:required_num_train_normal]
                    num_train_normal = len(train_normal_indices)
                    self.logger.debug(f"Changed normal training data to {num_train_normal}.")
        else:
            num_normal = len(self.train_normal_indices) + len(self.test_normal_indices)
            num_outlier = len(self.train_outlier_indices) + len(self.test_outlier_indices)
            contamination_ratio = num_outlier / (num_normal + num_outlier)
            self.logger.debug(f"Dataset is used with standard outlier fraction {contamination_ratio*100}%.")

        self.logger.debug(f"Using training data: {len(train_normal_indices)} normal "
                          f"and {len(train_outlier_indices)} anomaly data.")

        X_train = self._X[train_normal_indices+train_outlier_indices]
        y_train = self._y[train_normal_indices+train_outlier_indices].flatten()

        return X_train, y_train

    def get_test_data(self):
        X_test = self._X[self.test_normal_indices.tolist()+self.test_outlier_indices.tolist()]
        y_test = self._y[self.test_normal_indices.tolist()+self.test_outlier_indices.tolist()].flatten()

        return X_test, y_test

    def get_loader(self, X, y=None, train=True, batch_size=128):
        dataset = _OutlierDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(X)) if train else 1,
            shuffle=True if train else False,
            drop_last=True if train else False,
            num_workers=0,
        )

    def __repr__(self):
        return self.__class__.__name__ + f"_{self.rng.get_state()}"

    def get_name(self):
        return self.__class__.__name__.lower()


class _OutlierDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        if y is None:
            self.y = np.zeros((self.X.shape[0], 1), dtype=int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X = X.reshape(1, -1)
        X = X[0]

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(np.array([self.y[idx]]))

        return X, y
