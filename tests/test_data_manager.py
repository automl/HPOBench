import shutil
from multiprocessing import Pool

import pytest

import hpobench
from hpobench.util.data_manager import NASBench_201Data, YearPredictionMSDData, ProteinStructureData, BostonHousingData
skip_message = 'We currently skip this test because it takes too much time.'


@pytest.mark.skip(reason=skip_message)
def test_nasbench_201_load_thread_safe():
    shutil.rmtree(hpobench.config_file.data_dir / "nasbench_201", ignore_errors=True)
    function = lambda: NASBench_201Data(dataset='cifar100').load()
    with Pool(3) as pool:
        pool.map(function, [])


@pytest.mark.skip(reason=skip_message)
def test_nasbench_201_init():

    data_manager = NASBench_201Data(dataset='cifar100')
    assert len(data_manager.files) == 3
    assert all([file.startswith('NAS-Bench') for file in data_manager.files])

    with pytest.raises(AssertionError):
        NASBench_201Data(dataset='Non_existing_dataset')

    assert data_manager._save_dir == hpobench.config_file.data_dir / "nasbench_201"
    assert data_manager._save_dir.exists()


@pytest.mark.skip(reason=skip_message)
def test_nasbench_201_load():

    shutil.rmtree(hpobench.config_file.data_dir / "nasbench_201", ignore_errors=True)

    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()

    assert len(data) == 3
    assert (hpobench.config_file.data_dir / "nasbench_201").exists()
    assert len(list((hpobench.config_file.data_dir / "nasbench_201").glob('*.json'))) == 3
    assert not (hpobench.config_file.data_dir / "nasbench_201_data_v1.3.zip").exists()

    data_manager.data = None

    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()
    assert len(data) == 3


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


def test_tabular_datamanager():
    from hpobench.util.data_manager import TabularDataManager
    dm = TabularDataManager(model='lr',
                            task_id='3')

    table, meta_data = dm.load()

    assert (hpobench.config_file.data_dir / "TabularData" / 'lr' / str(3) / f'lr_3_data.parquet.gzip').exists()
    assert (hpobench.config_file.data_dir / "TabularData" / 'lr' / str(3) / f'lr_3_metadata.json').exists()

    table_2, meta_data_2 = dm.load()
