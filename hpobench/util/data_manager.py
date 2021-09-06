""" DataManager organizing the data for the benchmarks.

DataManager organizing the download of the data. Each data set should have an
own DataManger. The load function of a DataManger downloads the data from a
given online source and splits the data train, test and optional validation
splits.

For OpenML data sets (defined by task id or similar) please use the
hpobench.util.openml_data_manager.
"""

# pylint: disable=logging-fstring-interpolation,invalid-name

import abc
import gzip
import json
import logging
import pickle
import tarfile
from io import BytesIO
from pathlib import Path
from time import time
from typing import Tuple, Dict, Any, Union
from urllib.request import urlretrieve, urlopen
from zipfile import ZipFile

import numpy as np
import requests

try:
    from oslo_concurrency import lockutils
except ImportError:
    print("oslo_concurrency not installed, can't download datasets for nasbench201 (not needed for containers)")

try:
    import pandas as pd
except ImportError:
    print("pandas is not installed, can't download datasets for the ml.tabular_benchmarks (not needed for containers)")


import hpobench


class DataManager(abc.ABC, metaclass=abc.ABCMeta):
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

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_download_file', delay=0.5)
    def _download_file_with_progressbar(self, data_url: str, data_file: Path):
        data_file = Path(data_file)

        if data_file.exists():
            self.logger.info('Data File already exists. Skip downloading.')
            return

        self.logger.info(f"Download the file from {data_url} to {data_file}")
        data_file.parent.mkdir(parents=True, exist_ok=True)

        from tqdm import tqdm
        r = requests.get(data_url, stream=True)
        with open(data_file, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                              unit_divisor=1024, unit='kB', total=int(total_length / 1024) + 1):
                if chunk:
                    _ = f.write(chunk)
                    f.flush()
        self.logger.info(f"Finished downloading to {data_file}")

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_unzip_file', delay=0.5)
    def _untar_data(self, compressed_file: Path, save_dir: Union[Path, None] = None):
        self.logger.debug('Extract the compressed data')
        with tarfile.open(compressed_file, 'r') as fh:
            if save_dir is None:
                save_dir = compressed_file.parent
            fh.extractall(save_dir)
        self.logger.debug(f'Successfully extracted the data to {save_dir}')

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_unzip_file', delay=0.5)
    def _unzip_data(self, compressed_file: Path, save_dir: Union[Path, None] = None):
        self.logger.debug('Extract the compressed data')
        with ZipFile(compressed_file, 'r') as fh:
            if save_dir is None:
                save_dir = compressed_file.parent
            fh.extractall(save_dir)
        self.logger.debug(f'Successfully extracted the data to {save_dir}')


class HoldoutDataManager(DataManager):
    """  Base Class for loading and managing the Holdout data sets.

    Attributes
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_valid : np.ndarray
    y_valid : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """

    def __init__(self):
        super().__init__()

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
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
        self._save_to = hpobench.config_file.data_dir / "MNIST"

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
        self._save_to = hpobench.config_file.data_dir / "cifar10"

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
            fh = open(self.__load_data(filename=f'data_batch_{j + 1}'), "rb")
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
        self._save_to = hpobench.config_file.data_dir / "svhn"

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

        return X_train, y_train[:, 0], X_valid, y_valid[:, 0], X_test, y_test[:, 0]

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

            # pylint: disable=import-outside-toplevel
            from scipy.io import loadmat
            data = loadmat(save_fl)

            x = data['X'].T
            y = data['y']
            return x, y

        X_train, y_train = __load_x_y(filename_train)
        X_test, y_test = __load_x_y(filename_test)

        return X_train, y_train, X_test, y_test


class NASBench_201Data(DataManager):
    """ Download the necessary files for the nasbench201 benchmark. The benchmark has a data file for every pair of
    data set (cifar10, cifar10-valid, cifar100, ImageNet16-120)
    seed (777,888,999)
    metric (train_acc1es, train_times, train_losses, eval_acc1es, eval_times, eval_losses)

    Download for each data set the all corresponding data files.
    The files should be hosted on automl.org.

    For more information about the metric, have a look in the benchmark docstrings.
    """

    def __init__(self, dataset: str):
        """
        Init the NasbenchData Manager.

        Parameters
        ----------
        dataset : str
            One of cifar10, cifar10-valid, cifar100, ImageNet16-120
        """
        all_datasets = ['cifar10-valid', 'cifar100', 'ImageNet16-120']
        assert dataset in all_datasets, f'data set {dataset} unknown'

        super(NASBench_201Data, self).__init__()

        self.files = [f'NAS-Bench-201-v1_1-096897_{dataset}.json' for dataset in all_datasets]
        self._save_dir = hpobench.config_file.data_dir / "nasbench_201"
        self.filename = f'NAS-Bench-201-v1_1-096897_{dataset}.json'

        self._url_source = 'https://www.automl.org/wp-content/uploads/2020/08/nasbench_201_data_v1.3.zip'
        self.data = {}

        self.create_save_directory(self._save_dir)

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_nasbench_201_data', delay=0.5)
    def _download(self):
        # Check if data is already downloaded. If a single file is missing, we have to download the complete zip again.
        # Use a file lock to ensure that no two processes try to download the same files at the same time.
        file_is_missing = not all([(self._save_dir / file).exists() for file in self.files])

        if not file_is_missing:
            self.logger.debug('NasBench201DataManager: Data already downloaded')
        else:
            self.logger.info(f'NasBench201DataManager: Start downloading data from {self._url_source} '
                             f'to {self._save_dir}')

            with urlopen(self._url_source) as zip_archive:
                with ZipFile(BytesIO(zip_archive.read())) as zip_file:
                    zip_file.extractall(self._save_dir)

    def _load(self) -> Dict:
        """ Load the data from the file system """
        import json

        with (self._save_dir / self.filename).open('rb') as fh:
            data = json.load(fh)

        return data

    def load(self) -> Dict:
        """ Loads data from data directory as defined in config_file.data_directory"""
        self.logger.debug('NasBench201DataManager: Starting to load data')
        t = time()

        self._download()
        self.data = self._load()
        self.logger.info(f'NasBench201DataManager: Data successfully loaded after {time() - t:.2f}')

        return self.data


class NASBench_101DataManager(DataManager):
    def __init__(self, data_path: Union[str, Path, None] = None):
        super(NASBench_101DataManager, self).__init__()

        self.save_dir = (hpobench.config_file.data_dir / "nasbench_101") if data_path is None else Path(data_path)
        self.fname = 'nasbench_full.tfrecord'
        self.url = 'https://storage.googleapis.com/nasbench/' + self.fname

        self.create_save_directory(self.save_dir)

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_nasbench_101_data', delay=0.5)
    def _download(self, save_to: Path):
        from tqdm import tqdm

        r = requests.get(self.url, stream=True)
        with save_to.open('wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                              unit_divisor=1024, unit='kB', total=int(total_length / 1024) + 1):
                if chunk:
                    _ = f.write(chunk)
                    f.flush()

    def download(self) -> None:
        """ This function downloads (if necessary) the api file. """
        if not (self.save_dir / self.fname).exists():
            self.logger.info(f'NasBench101DataManager: File {self.save_dir / self.fname} not found.'
                             f' Start downloading.')
            self._download(save_to=self.save_dir / self.fname)
        else:
            self.logger.info('NasBench101DataManager: Data already available. Skip downloading.')

    def load(self) -> Any:
        """ Loads data from data directory as defined in config_file.data_directory"""
        self.logger.debug('NasBench101DataManager: Starting to load data')
        t = time()

        self.download()

        from nasbench import api
        data = api.NASBench(str(self.save_dir / self.fname))
        self.logger.info(f'NasBench101DataManager: Data successfully loaded after {time() - t:.2f}')
        return data


class SurrogateDataManger(DataManager):
    def __init__(self, dataset: str):

        allowed_datasets = ["adult", "higgs", "letter", "mnist", "optdigits", "poker", "svm"]
        assert dataset in allowed_datasets, f'Requested data set is not supported. Must be one of ' \
                                            f'{", ".join(allowed_datasets)}, but was {dataset}'

        super(SurrogateDataManger, self).__init__()

        self.url_source = 'https://www.automl.org/wp-content/uploads/2019/05/surrogates.tar.gz'
        self.dataset = dataset
        self.save_dir = hpobench.config_file.data_dir / "Surrogates"
        self.compressed_data = self.save_dir / 'surrogates.tar.gz'
        self.obj_fn_file = None
        self.cost_file = None

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_surrogates_data', delay=0.5)
    def _check_availability_and_download(self):

        # Check if the compressed data file is already available. This check is moved in this function to ensure
        # that no process can detect this file, when it is still in the process of downloading and
        # think that it is already there.
        if self.compressed_data.exists():
            self.logger.info("Tar file found. Skip redownloading.")
            return

        self.logger.info("Tar file not found. Download the compressed data.")
        self.compressed_data.parent.mkdir(parents=True, exist_ok=True)

        from tqdm import tqdm
        r = requests.get(self.url_source, stream=True)
        with open(self.compressed_data, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                              unit_divisor=1024, unit='kB', total=int(total_length / 1024) + 1):
                if chunk:
                    _ = f.write(chunk)
                    f.flush()
        self.logger.info("Finished downloading")

    # pylint: disable=arguments-differ
    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_surrogates_unzip_data', delay=0.5)
    def _unzip_data(self):
        self.logger.debug('Extract the compressed data')
        with tarfile.open(self.compressed_data, 'r') as fh:
            fh.extractall(self.save_dir)
        self.logger.debug(f'Successfully extracted the data to {self.save_dir}')

    def load(self):
        self.logger.info(f"Start to load the data from {self.save_dir} for dataset {self.dataset}")

        assert self.obj_fn_file is not None
        assert self.cost_file is not None

        # Check if the surrogate files are already available
        if not (self.obj_fn_file.exists() or self.cost_file.exists()):
            self.logger.info(f"One of the files {self.obj_fn_file} and {self.cost_file} not found.")

            # If not, then check if we have to download the compressed data or if this file isn't already there,
            # download it again.
            self._check_availability_and_download()

            # Extract the compressed data
            self._unzip_data()

        self.logger.debug('Load the obj function values from file.')
        with open(self.obj_fn_file, 'rb') as fh:
            surrogate_objective = pickle.load(fh)

        self.logger.debug('Load the cost values from file.')
        with open(self.cost_file, 'rb') as fh:
            surrogate_costs = pickle.load(fh)

        self.logger.info(f'Finished loading the data for paramenet - dataset: {self.dataset}')
        return surrogate_objective, surrogate_costs


class ParamNetDataManager(SurrogateDataManger):
    def __init__(self, dataset: str):
        super(ParamNetDataManager, self).__init__(dataset)
        self.obj_fn_file = self.save_dir / f'rf_surrogate_paramnet_{dataset}.pkl'
        self.cost_file = self.save_dir / f'rf_cost_surrogate_paramnet_{dataset}.pkl'


class SurrogateSVMDataManager(SurrogateDataManger):
    def __init__(self):
        super(SurrogateSVMDataManager, self).__init__(dataset='svm')
        self.obj_fn_file = self.save_dir / 'rf_surrogate_svm.pkl'
        self.cost_file = self.save_dir / 'rf_cost_surrogate_svm.pkl'


class BostonHousingData(HoldoutDataManager):

    def __init__(self):

        super(BostonHousingData, self).__init__()
        self.url_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
        self._save_dir = hpobench.config_file.data_dir / "BostonHousing"
        self.create_save_directory(self._save_dir)

    def load(self):
        """
        Loads BostonHousing from data directory as defined in hpobenchrc.data_directory.
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
        self.logger.debug('BostonHousingDataManager: Starting to load data')
        t = time()

        self._download()

        X_trn, y_trn, X_val, y_val, X_tst, y_tst = self._load()
        self.logger.info(f'BostonHousingDataManager: Data successfully loaded after {time() - t:.2f}')

        return X_trn, y_trn, X_val, y_val, X_tst, y_tst

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_protein_structure_data', delay=0.5)
    def _download(self):
        """
        Loads data from UCI website
        https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
        If necessary downloads data, otherwise loads data from data_directory
        """
        # Check if data is already downloaded.
        # Use a file lock to ensure that no two processes try to download the same files at the same time.
        if (self._save_dir / 'housing.data').exists():
            self.logger.debug('BostonHousingDataManager: Data already downloaded')
        else:
            self.logger.info(f'BostonHousingDataManager: Start downloading data from {self.url_source} '
                             f'to {self._save_dir}')
            urlretrieve(self.url_source, self._save_dir / 'housing.data')

    def _load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the data from file and split it into train, test and validation split.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """
        data = np.loadtxt(self._save_dir / 'housing.data')

        N = data.shape[0]

        n_train = int(N * 0.6)
        n_val = int(N * 0.2)

        X_train, y_train = data[:n_train, :-1], data[:n_train, -1]
        X_val, y_val = data[n_train:n_train + n_val, :-1], data[n_train:n_train + n_val, -1]
        X_test, y_test = data[n_train + n_val:, :-1], data[n_train + n_val:, -1]

        return X_train, y_train, X_val, y_val, X_test, y_test


class ProteinStructureData(HoldoutDataManager):

    def __init__(self):
        super(ProteinStructureData, self).__init__()
        self.url_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'
        self._save_dir = hpobench.config_file.data_dir / "ProteinStructure"
        self.create_save_directory(self._save_dir)

    def load(self):
        """
        Loads Physicochemical Properties of Protein Tertiary Structure Data Set
        from data directory as defined in _config.data_directory.
        Downloads data if necessary from UCI.
        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """
        self.logger.debug('ProteinStructureDataManager: Starting to load data')
        t = time()
        self._download()

        X_train, y_train, X_val, y_val, X_test, y_test = self._load()
        self.logger.info(f'ProteinStructureDataManager: Data successfully loaded after {time() - t:.2f}')

        return X_train, y_train, X_val, y_val, X_test, y_test

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_protein_structure_data', delay=0.5)
    def _download(self):
        """
        Loads data from UCI website
        https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv
        If necessary downloads data, otherwise loads data from data_directory
        """
        # Check if data is already downloaded.
        # Use a file lock to ensure that no two processes try to download the same files at the same time.
        if (self._save_dir / 'CASP.csv').exists():
            self.logger.debug('ProteinStructureDataManager: Data already downloaded')
        else:
            self.logger.info(f'ProteinStructureDataManager: Start downloading data from {self.url_source} '
                             f'to {self._save_dir}')
            urlretrieve(self.url_source, self._save_dir / 'CASP.csv')

    def _load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the data from file and split it into train, test and validation split.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """
        data = np.loadtxt(self._save_dir / 'CASP.csv', delimiter=',', skiprows=1)

        n_train = int(data.shape[0] * 0.6)
        n_val = int(data.shape[0] * 0.2)

        # note the target value is the first column for this dataset!
        X_train, y_train = data[:n_train, 1:], data[:n_train, 0]
        X_val, y_val = data[n_train:n_train + n_val, 1:], data[n_train:n_train + n_val, 0]
        X_test, y_test = data[n_train + n_val:, 1:], data[n_train + n_val:, 0]

        return X_train, y_train, X_val, y_val, X_test, y_test


class YearPredictionMSDData(HoldoutDataManager):

    def __init__(self):
        super(YearPredictionMSDData, self).__init__()
        self.url_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
        self._save_dir = hpobench.config_file.data_dir / "YearPredictionMSD"
        self.create_save_directory(self._save_dir)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads Physicochemical Properties of Protein Tertiary Structure Data Set from data directory as
        defined in _config.data_directory. Downloads data if necessary from UCI.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """
        self.logger.debug('YearPredictionMSDDataManager: Starting to load data')
        t = time()

        self._download()

        X_trn, y_trn, X_val, y_val, X_tst, y_tst = self._load()
        self.logger.info(f'YearPredictionMSDDataManager: Data successfully loaded after {time() - t:.2f}')

        return X_trn, y_trn, X_val, y_val, X_tst, y_tst

    @lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{hpobench.config_file.cache_dir}/lock_year_prediction_data', delay=0.5)
    def _download(self):
        # Check if data is already downloaded.
        # Use a file lock to ensure that no two processes try to download the same files at the same time.

        if (self._save_dir / 'YearPredictionMSD.txt').exists():
            self.logger.debug('YearPredictionMSDDataManager: Data already downloaded')
        else:
            self.logger.info(f'YearPredictionMSDDataManager: Start downloading data from {self.url_source} '
                             f'to {self._save_dir}')

            with urlopen(self.url_source) as zip_archive:
                with ZipFile(BytesIO(zip_archive.read())) as zip_file:
                    zip_file.extractall(self._save_dir)

    def _load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the data from file and split it into train, test and validation split.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        """

        with (self._save_dir / 'YearPredictionMSD.txt').open('r') as fh:
            data = np.loadtxt(fh, delimiter=',')

        # Use 70% of the data as train split, 20% as validation split and 10% as test split
        n_trn = int(data.shape[0] * 0.7)
        n_val = int(data.shape[0] * 0.2)

        # Note the target value is the first column for this dataset!
        X_trn, y_trn = data[:n_trn, 1:], data[:n_trn, 0]
        X_val, y_val = data[n_trn:n_trn + n_val, 1:], data[n_trn:n_trn + n_val, 0]
        X_tst, y_tst = data[n_trn + n_val:, 1:], data[n_trn + n_val:, 0]

        return X_trn, y_trn, X_val, y_val, X_tst, y_tst


class TabularDataManager(DataManager):
    def __init__(self, model: str, task_id: [int, str], data_dir: [str, Path, None] = None):
        super(TabularDataManager, self).__init__()

        url_dict = dict(
            xgb="https://ndownloader.figshare.com/files/30469920",
            svm="https://ndownloader.figshare.com/files/30379359",
            lr="https://ndownloader.figshare.com/files/30379038",
            rf="https://ndownloader.figshare.com/files/30469089",
            nn="https://ndownloader.figshare.com/files/30379005"
        )

        assert model in url_dict.keys(), \
            f'Model has to be one of {list(url_dict.keys())} but was {model}'

        self.model = model
        self.task_id = str(task_id)

        self.url_to_use = url_dict.get(model)

        if data_dir is None:
            data_dir = hpobench.config_file.data_dir / "TabularData"

        self._save_dir = Path(data_dir) / self.model
        self.create_save_directory(self._save_dir)

        self.parquet_file = self._save_dir / self.task_id / f'{self.model}_{self.task_id}_data.parquet.gzip'
        self.metadata_file = self._save_dir / self.task_id / f'{self.model}_{self.task_id}_metadata.json'

    # pylint: disable=arguments-differ
    def load(self):
        # Can we directly load the files?
        if self.parquet_file.exists() and self.metadata_file.exists():
            table = self._load_parquet(self.parquet_file)
            metadata = self._load_json(self.metadata_file)
            return table, metadata

        # We have to download the entire zip file and etract then extract the parquet file.
        self._download_file_with_progressbar(self.url_to_use, self._save_dir / f'{self.model}.zip')
        self._unzip_data(self._save_dir / f'{self.model}.zip', self._save_dir)
        table = self._load_parquet(self.parquet_file)
        metadata = self._load_json(self.metadata_file)
        return table, metadata

    @staticmethod
    def _load_parquet(path):
        data = pd.read_parquet(path)
        return data

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
