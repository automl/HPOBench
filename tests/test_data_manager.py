import pytest

import hpolib


def test_nasbench_201_get_files():
    from hpolib.util.data_manager import NASBench_201Data

    files = NASBench_201Data.get_files_per_dataset(dataset='cifar10')
    assert len(files) == 18
    assert all([file.startswith('nb201_cifar10') for file in files])


def test_nasbench_201_get_metrics():
    from hpolib.util.data_manager import NASBench_201Data

    metrics = NASBench_201Data.get_metrics()
    assert metrics == ['train_acc1es', 'train_losses', 'train_times',
                       'eval_acc1es', 'eval_times', 'eval_losses']


def test_nasbench_201_init():
    from hpolib.util.data_manager import NASBench_201Data

    data_manager = NASBench_201Data(dataset='cifar100')
    assert len(data_manager.files) == 18
    assert all([file.startswith('nb201_cifar10') for file in data_manager.files])

    with pytest.raises(AssertionError):
        NASBench_201Data(dataset='Non_existing_dataset')

    assert data_manager._save_dir == hpolib.config_file.data_dir / "nasbench_201"


def test_nasbench_201_load():
    from hpolib.util.data_manager import NASBench_201Data
    data_manager = NASBench_201Data(dataset='cifar100')
    data = data_manager.load()
    assert data is not None
    assert len(data) == 3 * len(NASBench_201Data.get_metrics())
