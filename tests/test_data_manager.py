import shutil
from multiprocessing import Pool

import pytest

import hpobench
from hpobench.util.data_manager import NASBench_201Data, YearPredictionMSDData, ProteinStructureData, BostonHousingData


def test_nasbench_201_load_thread_safe():
    shutil.rmtree(hpobench.config_file.data_dir / "nasbench_201", ignore_errors=True)
    function = lambda: NASBench_201Data(dataset='cifar100').load()
    with Pool(3) as pool:
        pool.map(function, [])


def test_nasbench_201_get_files():

    files = NASBench_201Data.get_files_per_dataset(dataset='cifar10')
    assert len(files) == 27
    assert all([file.startswith('nb201_cifar10') for file in files])


def test_nasbench_201_get_metrics():

    metrics = NASBench_201Data.get_metrics()
    assert metrics == ['train_acc1es', 'train_losses', 'train_times',
                       'valid_acc1es', 'valid_times', 'valid_losses',
                       'test_acc1es', 'test_times', 'test_losses']


def test_nasbench_201_init():

    data_manager = NASBench_201Data(dataset='cifar100')
    assert len(data_manager.files) == 27
    assert all([file.startswith('nb201_cifar10') for file in data_manager.files])

    with pytest.raises(AssertionError):
        NASBench_201Data(dataset='Non_existing_dataset')

    assert data_manager._save_dir == hpobench.config_file.data_dir / "nasbench_201"
    assert data_manager._save_dir.exists()


def test_nasbench_201_load():

    shutil.rmtree(hpobench.config_file.data_dir / "nasbench_201", ignore_errors=True)

    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()

    assert len(data) == len(list(NASBench_201Data.get_seeds_metrics()))
    assert len(data) == 3 * len(NASBench_201Data.get_metrics())
    assert (hpobench.config_file.data_dir / "nasbench_201").exists()
    assert len(list((hpobench.config_file.data_dir / "nasbench_201").glob('*.pkl'))) == 72
    assert not (hpobench.config_file.data_dir / "nasbench_201_data_v1.2.zip").exists()

    data_manager.data = None

    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()
    assert len(data) == 3 * len(NASBench_201Data.get_metrics())


def test_year_prediction_msd_data():
    dm = YearPredictionMSDData()
    assert dm.url_source == 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
    assert dm._save_dir.exists()

    # First one downloads the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = dm.load()

    # second call should check the 'if exists' branch
    _ = dm.load()

    # train = 70%, valid = 20%, test = 10%
    assert 0 < len(x_test) == len(y_test)
    assert len(y_test) < len(x_valid) == len(y_valid)
    assert len(y_valid) < len(x_train) == len(y_train)


def test_protein_structure_data():
    dm = ProteinStructureData()
    assert dm.url_source == 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'
    assert dm._save_dir.exists()

    # First one downloads the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = dm.load()

    # second call should check the 'if exists' branch
    _ = dm.load()

    # train = 60%, valid = 20%, test = 20%
    assert 0 < len(x_test) == len(y_test)
    assert 0 < len(x_valid) == len(y_valid)
    assert len(y_valid) < len(x_train) == len(y_train)


def test_boston_data():
    dm = BostonHousingData()
    assert dm.url_source == 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    assert dm._save_dir.exists()

    # First one downloads the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = dm.load()

    # second call should check the 'if exists' branch
    _ = dm.load()

    # train = 60%, valid = 20%, test = 20%
    assert 0 < len(x_test) == len(y_test)
    assert 0 < len(x_valid) == len(y_valid)
    assert len(y_valid) < len(x_train) == len(y_train)
