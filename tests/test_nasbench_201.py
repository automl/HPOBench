import logging

import pytest

logging.basicConfig(level=logging.DEBUG)

import os

os.environ['HPOLIB_DEBUG'] = 'true'

from hpolib.container.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark, Cifar100NasBench201Benchmark, \
    Cifar10ValidNasBench201Benchmark, Cifar10NasBench201Benchmark as Cifar10NasBench201BenchmarkContainer

from hpolib.benchmarks.nas.nasbench_201 import Cifar10NasBench201Benchmark


def test_nasbench201_cifar10valid():
    b = Cifar10ValidNasBench201Benchmark(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result['function_value'] == pytest.approx(0.411, abs=0.1)
    assert result['cost'] == pytest.approx(2205.87, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']
    assert result['info']['train_cost'] == result['cost']


def test_nasbench201_cifar100():
    b = Cifar100NasBench201Benchmark(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result is not None
    assert result['function_value'] == pytest.approx(7.8259, abs=0.1)
    assert result['cost'] == pytest.approx(4411.75, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']
    assert result['info']['train_cost'] == result['cost']


def test_nasbench201_Image():
    b = ImageNetNasBench201Benchmark(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result is not None
    assert result['function_value'] == pytest.approx(62.858, abs=0.1)
    assert result['cost'] == pytest.approx(13385.25, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']
    assert result['info']['train_cost'] == result['cost']

def test_nasbench201_cifar10_container():
    b = Cifar10NasBench201BenchmarkContainer(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result is not None
    assert result['function_value'] == pytest.approx(0.5019, abs=0.1)
    assert result['cost'] == pytest.approx(4411.75, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']


def test_nasbench201_cifar10():
    b = Cifar10NasBench201Benchmark(rng=0)

    assert b.data is not None
    assert len(b.get_meta_information()) == 2

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result is not None
    assert result['function_value'] == pytest.approx(0.5019, abs=0.1)
    assert result['cost'] == pytest.approx(4411.75, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']

    result_test = b.objective_function_test(configuration=config, fidelity=fidelity)
    assert result['info']['train_precision'] == result_test['info']['train_precision']
    assert result['info']['train_cost'] == result_test['info']['train_cost']
    assert result['info']['train_losses'] == result_test['info']['train_losses']
    assert result['info']['eval_precision'] == result_test['info']['eval_precision']
    assert result['info']['eval_losses'] == result_test['info']['eval_losses']
    assert result['info']['eval_cost'] == result_test['info']['eval_cost']

    assert result_test['cost'] > result['cost']

    result_lower = b.objective_function(configuration=config, fidelity={'epoch': 100},
                                        data_seed=(777, 888, 999))
    assert result['cost'] > result_lower['cost']

    with pytest.raises(ValueError):
        b.objective_function(configuration=config, fidelity={'epoch': 200}, data_seed=0.1)

    with pytest.raises(AssertionError):
        b.objective_function(configuration=config, fidelity=fidelity, data_seed=0.1)

    with pytest.raises(AssertionError):
        b.objective_function(configuration=config, fidelity=fidelity, data_seed=0)

    with pytest.raises(AssertionError):
        b.objective_function(configuration=config, fidelity=fidelity, data_seed=[777])

    with pytest.raises(AssertionError):
        b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 881))


def test_nasbench201_fidelity_space():
    fs = Cifar10NasBench201Benchmark(rng=0).get_fidelity_space()
    assert len(fs.get_hyperparameters()) == 1


def test_nasbench201_config():
    cs = Cifar10NasBench201Benchmark(rng=0).get_configuration_space(0)
    c = cs.sample_configuration()
    func = Cifar10NasBench201Benchmark.config_to_structure_func(4)
    struct = func(c)

    assert struct.__repr__() == '_Structure(4 nodes with |avg_pool_3x3~0|+|none~0|nor_conv_3x3~1|+' \
                                '|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|)'
    assert len(struct) == 4
    assert struct[0] == (('avg_pool_3x3', 0),)

    struct_str = struct.tostr()
    assert struct_str == '|avg_pool_3x3~0|+|none~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|'
