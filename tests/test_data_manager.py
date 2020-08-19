import pytest
import hpolib
from hpolib.util.data_manager import NASBench_201Data
import shutil
from multiprocessing import Pool


def test_nasbench_201_load_thread_safe():
    shutil.rmtree(hpolib.config_file.data_dir / "nasbench_201", ignore_errors=True)
    function = lambda: NASBench_201Data(dataset='cifar100').load()
    with Pool(3) as pool:
        pool.map(function, [])


def test_nasbench_201_get_files():

    files = NASBench_201Data.get_files_per_dataset(dataset='cifar10')
    assert len(files) == 18
    assert all([file.startswith('nb201_cifar10') for file in files])


def test_nasbench_201_get_metrics():

    metrics = NASBench_201Data.get_metrics()
    assert metrics == ['train_acc1es', 'train_losses', 'train_times',
                       'eval_acc1es', 'eval_times', 'eval_losses']


def test_nasbench_201_init():

    data_manager = NASBench_201Data(dataset='cifar100')
    assert len(data_manager.files) == 18
    assert all([file.startswith('nb201_cifar10') for file in data_manager.files])

    with pytest.raises(AssertionError):
        NASBench_201Data(dataset='Non_existing_dataset')

    assert data_manager._save_dir == hpolib.config_file.data_dir / "nasbench_201"
    assert data_manager._save_dir.exists()


def test_nasbench_201_load():

    shutil.rmtree(hpolib.config_file.data_dir / "nasbench_201", ignore_errors=True)

    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()

    assert len(data) == len(list(NASBench_201Data.get_seeds_metrics()))
    assert len(data) == 3 * len(NASBench_201Data.get_metrics())
    assert (hpolib.config_file.data_dir / "nasbench_201").exists()
    assert len(list((hpolib.config_file.data_dir / "nasbench_201" / "data").glob('*.pkl'))) == 72
    assert not (hpolib.config_file.data_dir / "nasbench_201_data_v1.1.zip").exists()

    data_manager.data = None

    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()
    assert len(data) == 3 * len(NASBench_201Data.get_metrics())
