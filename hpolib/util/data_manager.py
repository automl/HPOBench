""" DataManager organizing the data for the benchmarks.

DataManager organizing the download of the data. Each data set should have an
own DataManger. The load function of a DataManger downloads the data from a
given online source and splits the data train, test and optional validation
splits.

For OpenML data sets (defined by task id or similar) please use the
hpolib.util.openml_data_manager.
"""

import abc
import gzip
import logging
import pickle
import tarfile
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
from scipy.io import loadmat

import hpolib


class DataManager(object, metaclass=abc.ABCMeta):
    """ Base Class for loading and managing the data.

    Attributes
    ----------
    logger : logging.Logger

    """

    def __init__(self):
        self.logger = logging.getLogger("DataManager")

    @abc.abstractmethod
    def load(self):
        """ Loads data from data directory as defined in
        config_file.data_directory
        """
        raise NotImplementedError()

    def create_save_directory(self, save_dir: Path):
        """ Helper function. Check if data directory exists. If not, create it.

        Parameters
        ----------
        save_dir : Path
            Path to the directory. where the data should be stored
        """
        if not save_dir.is_dir():
            self.logger.debug(f'Create directory {save_dir}')
            save_dir.mkdir(parents=True, exist_ok=True)


class HoldoutDataManager(DataManager):
    """  Base Class for loading and managing the Holdout data sets.

    Attributes
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_val : np.ndarray
    y_val : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """

    def __init__(self):
        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None


class CrossvalidationDataManager(DataManager):
    """
    Base Class for loading and managing the cross-validation data sets.

    Attributes
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """

    def __init__(self):
        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


class MNISTData(HoldoutDataManager):
    """Class implementing the HoldoutDataManger, managing the MNIST data set"""

    def __init__(self):
        super(MNISTData, self).__init__()

        self._url_source = 'http://yann.lecun.com/exdb/mnist/'
        self._save_to = hpolib.config_file.data_dir / "MNIST"

        self.create_save_directory(self._save_to)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads MNIST from data directory as defined in
        config_file.data_directory. Downloads data if necessary. Code is copied
         and modified from the Lasagne tutorial.

        Returns
        -------
        X_train : np.ndarray
        y_train : np.ndarray
        X_val : np.ndarray
        y_val : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        X_train = self.__load_data(filename='train-images-idx3-ubyte.gz',
                                   images=True)
        y_train = self.__load_data(filename='train-labels-idx1-ubyte.gz')
        X_test = self.__load_data(filename='t10k-images-idx3-ubyte.gz',
                                  images=True)
        y_test = self.__load_data(filename='t10k-labels-idx1-ubyte.gz')

        # Split data in training and validation
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        assert X_train.shape[0] == 50000, X_train.shape
        assert X_val.shape[0] == 10000, X_val.shape
        assert X_test.shape[0] == 10000, X_test.shape

        # Reshape data to NxD
        X_train = X_train.reshape(X_train.shape[0], 28 * 28)
        X_val = X_val.reshape(X_val.shape[0], 28 * 28)
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def __load_data(self, filename: str, images: bool = False) -> np.ndarray:
        """
        Loads data in Yann LeCun's binary format as available under
        'http://yann.lecun.com/exdb/mnist/'.
        If necessary downloads data, otherwise loads data from data_directory

        Parameters
        ----------
        filename : str
            file to download
        images : bool
            if True converts data to X

        Returns
        -------
        np.ndarray
        """

        # 1) If necessary download data
        save_fl = self._save_to / filename
        if not save_fl.exists():
            self.logger.debug(f"Downloading {self._url_source + filename} "
                              f"to {save_fl}")
            urlretrieve(self._url_source + filename, str(save_fl))
        else:
            self.logger.debug(f"Load data {save_fl}")

        # 2) Read in data
        if images:
            with gzip.open(save_fl, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            # Follow the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # Convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at: http://deeplearning.net/data/mnist/mnist.pkl.gz.
            data = data / np.float32(256)
        else:
            with gzip.open(save_fl, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data


class MNISTDataCrossvalidation(MNISTData, CrossvalidationDataManager):
    """ Class loading the MNIST data set. """

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads MNIST from data directory as defined in
        config_file.data_directory. Downloads data if necessary.

        Returns
        -------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray

        """
        X_train, y_train, X_val, y_val, X_test, y_test = \
            super(MNISTDataCrossvalidation, self).load()

        X_train = np.concatenate([X_train, X_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)

        return X_train, y_train, X_test, y_test


class CIFAR10Data(DataManager):
    """ Class loading the Cifar10 data set. """

    def __init__(self):
        super(CIFAR10Data, self).__init__()

        self._url_source = 'https://www.cs.toronto.edu/~kriz/' \
                           'cifar-10-python.tar.gz'
        self._save_to = hpolib.config_file.data_dir / "cifar10"

        self.create_save_directory(self._save_to)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads CIFAR10 from data directory as defined in
        config_file.data_directory. Downloads data if necessary.

        Returns
        -------
        X_train : np.ndarray
        y_train : np.ndarray
        X_val : np.ndarray
        y_val : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """

        xs = []
        ys = []
        for j in range(5):
            fh = open(self.__load_data(filename=f'data_batch_{j+1}'), "rb")
            d = pickle.load(fh, encoding='latin1')
            fh.close()
            x = d['data']
            y = d['labels']
            xs.append(x)
            ys.append(y)

        fh = open(self.__load_data(filename='test_batch'), "rb")
        d = pickle.load(fh, encoding='latin1')
        fh.close()

        xs.append(d['data'])
        ys.append(d['labels'])

        x = np.concatenate(xs) / np.float32(255)
        y = np.concatenate(ys)
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

        # Subtract per-pixel mean
        pixel_mean = np.mean(x[0:50000], axis=0)

        x -= pixel_mean

        # Split in training, validation and test
        X_train = x[:40000, :, :, :]
        y_train = y[:40000]

        X_valid = x[40000:50000, :, :, :]
        y_valid = y[40000:50000]

        X_test = x[50000:, :, :, :]
        y_test = y[50000:]

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def __load_data(self, filename: str) -> Path:
        """
        Loads data in binary format as available under
        'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'.

        Parameters
        ----------
        filename : str
            file to download

        Returns
        -------
        Path
        """

        save_fl = self._save_to / 'cifar-10-batches-py' / filename
        if not save_fl.exists():
            self.logger.debug(f'Downloading {self._url_source} to {save_fl}')
            urlretrieve(self._url_source,
                        self._save_to / "cifar-10-python.tar.gz")
            tar = tarfile.open(self._save_to / "cifar-10-python.tar.gz")
            tar.extractall(self._save_to)

        else:
            self.logger.debug("Load data %s", save_fl)

        return save_fl


class SVHNData(DataManager):
    """ Class loading the house numbers data set.

    Attributes
    ----------
    n_train_all : int
    n_valid : int
    n_train : int
    n_test : int
    """

    def __init__(self):
        super(SVHNData, self).__init__()

        self._url_source = 'http://ufldl.stanford.edu/housenumbers/'
        self._save_to = hpolib.config_file.data_dir / "svhn"

        self.n_train_all = 73257
        self.n_valid = 6000
        self.n_train = self.n_train_all - self.n_valid
        self.n_test = 26032

        self.create_save_directory(self._save_to)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads SVHN from data directory as defined in
        config_file.data_directory. Downloads data if necessary.

        Returns
        -------
        X_train : np.ndarray
        y_train : np.ndarray
        X_val : np.ndarray
        y_val : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        X, y, X_test, y_test = self.__load_data("train_32x32.mat",
                                                "test_32x32.mat")

        # Change the label encoding from [1, ... 10] to [0, ..., 9]
        y = y - 1
        y_test = y_test - 1

        X_train = X[:self.n_train, :, :, :]
        y_train = y[:self.n_train]
        X_valid = X[self.n_train:self.n_train_all, :, :, :]
        y_valid = y[self.n_train:self.n_train_all]

        X_train = np.array(X_train, dtype=np.float32)
        X_valid = np.array(X_valid, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)

        all_X = [X_train, X_valid, X_test]

        # Subtract per pixel mean
        for X in all_X:
            data_shape = X.shape
            X = X.reshape(X.shape[0], np.product(X.shape[1:]))
            X -= X.mean(axis=1)[:, np.newaxis]
            X = X.reshape(data_shape)

        return X_train, y_train[:, 0], X_valid, y_valid[:, 0], \
            X_test, y_test[:, 0]

    def __load_data(self, filename_train: str,
                    filename_test: str) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
        """
        Loads data in binary format as available under
        'http://ufldl.stanford.edu/housenumbers/'.

        Parameters
        ----------
        filename_train : str
            file to download
        filename_test : str
            file to download

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray, np.ndarray]
        """

        def __load_x_y(file_name):
            save_fl = self._save_to / file_name
            if not save_fl.exists():
                self.logger.debug(f"Downloading {self._url_source + file_name}"
                                  f" to {save_fl}")
                urlretrieve(self._url_source + file_name, save_fl)
            else:
                self.logger.debug(f"Load data {save_fl}")

            data = loadmat(save_fl)

            x = data['X'].T
            y = data['y']
            return x, y

        X_train, y_train = __load_x_y(filename_train)
        X_test, y_test = __load_x_y(filename_test)

        return X_train, y_train, X_test, y_test
