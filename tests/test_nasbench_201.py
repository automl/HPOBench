import logging
logging.basicConfig(level=logging.DEBUG)

import pytest

from hpobench.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark, Cifar100NasBench201Benchmark, \
    Cifar10ValidNasBench201Benchmark

from hpobench.util.container_utils import disable_container_debug, enable_container_debug

skip_message = 'We currently skip this test because it takes too much time.'


@pytest.fixture(scope='module')
def enable_debug():
    enable_container_debug()
    yield
    disable_container_debug()


@pytest.mark.skip(reason=skip_message)
def test_nasbench201_cifar10valid(enable_debug):

    b = Cifar10ValidNasBench201Benchmark(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result['function_value'] == pytest.approx(0.411, abs=0.1)
    assert result['cost'] == pytest.approx(6650.88, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']
    assert result['info']['train_cost'] == result['cost']

    result = b.objective_function_test(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    with pytest.raises(AssertionError):
        result = b.objective_function_test(configuration=config, fidelity={'epoch': 10})

@pytest.mark.skip(reason=skip_message)
def test_nasbench201_cifar100(enable_debug):
    b = Cifar100NasBench201Benchmark(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result is not None
    assert result['function_value'] == pytest.approx(7.8259, abs=0.1)
    assert result['cost'] == pytest.approx(13301.76, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']
    assert result['info']['train_cost'] == result['cost']


@pytest.mark.skip(reason=skip_message)
def test_nasbench201_Image(enable_debug):
    b = ImageNetNasBench201Benchmark(rng=0)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration()
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))

    assert result is not None
    assert result['function_value'] == pytest.approx(62.858, abs=0.1)
    assert result['cost'] == pytest.approx(40357.56, abs=0.1)
    assert result['info']['train_precision'] == result['function_value']
    assert result['info']['train_cost'] == result['cost']


def test_nasbench201_fidelity_space():
    fs = Cifar10ValidNasBench201Benchmark.get_fidelity_space()
    assert len(fs.get_hyperparameters()) == 1


def test_nasbench201_config():
    cs = Cifar10ValidNasBench201Benchmark.get_configuration_space(seed=0)
    c = cs.sample_configuration()
    func = Cifar10ValidNasBench201Benchmark.config_to_structure_func(4)
    struct = func(c)

    assert struct.__repr__() == '_Structure(4 nodes with |avg_pool_3x3~0|+|none~0|nor_conv_3x3~1|+' \
                                '|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|)'
    assert len(struct) == 4
    assert struct[0] == (('avg_pool_3x3', 0),)

    struct_str = struct.tostr()
    assert struct_str == '|avg_pool_3x3~0|+|none~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|skip_connect~2|'
