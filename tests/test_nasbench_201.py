import logging
logging.basicConfig(level=logging.DEBUG)
import pytest

from hpobench.container.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark, Cifar100NasBench201Benchmark, \
    Cifar10ValidNasBench201Benchmark
from hpobench.benchmarks.nas.nasbench_201 import \
    Cifar10ValidNasBench201MOBenchmark as LocalCifar10ValidNasBench201MOBenchmark
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

    cs_1 = b.get_configuration_space(seed=0)
    config_1 = cs_1.sample_configuration()
    cs_2 = b.get_configuration_space(seed=0)
    config_2 = cs_2.sample_configuration()
    assert config_1 == config_2

    config = {
        '1<-0': 'nor_conv_1x1',
        '2<-0': 'nor_conv_3x3',
        '2<-1': 'nor_conv_3x3',
        '3<-0': 'nor_conv_1x1',
        '3<-1': 'nor_conv_1x1',
        '3<-2': 'nor_conv_3x3'
    }
    result = b.objective_function(configuration=config, fidelity={'epoch': 199}, data_seed=(777, 888, 999))
    assert result['function_value'] == pytest.approx(9.78, abs=0.1)
    assert result['cost'] == pytest.approx(11973.20, abs=0.1)
    assert result['info']['valid_precision'] == result['function_value']
    assert result['info']['valid_cost'] == result['cost']

    result = b.objective_function_test(configuration=config, fidelity={'epoch': 200})
    assert result['function_value'] == pytest.approx(9.70, abs=0.1)
    assert result['cost'] == pytest.approx(10426.33, abs=0.2)
    assert result['info']['test_precision'] == result['function_value']
    assert result['info']['test_cost'] == result['cost']

    with pytest.raises(ValueError):
        result = b.objective_function_test(configuration=config, fidelity={'epoch': 10})


@pytest.mark.skip(reason=skip_message)
def test_nasbench201_cifar100(enable_debug):
    b = Cifar100NasBench201Benchmark(rng=0)

    config = {'1<-0': 'nor_conv_1x1',
              '2<-0': 'nor_conv_3x3',
              '2<-1': 'nor_conv_3x3',
              '3<-0': 'nor_conv_1x1',
              '3<-1': 'nor_conv_1x1',
              '3<-2': 'nor_conv_3x3'}
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))
    assert result is not None
    assert result['function_value'] == pytest.approx(29.5233, abs=0.1)
    assert result['cost'] == pytest.approx(19681.70, abs=0.1)
    assert result['info']['valid_precision'] == result['function_value']
    assert result['info']['valid_cost'] == result['cost']


@pytest.mark.skip(reason=skip_message)
def test_nasbench201_Image(enable_debug):
    b = ImageNetNasBench201Benchmark(rng=0)
    config = {'1<-0': 'nor_conv_1x1',
              '2<-0': 'nor_conv_3x3',
              '2<-1': 'nor_conv_3x3',
              '3<-0': 'nor_conv_1x1',
              '3<-1': 'nor_conv_1x1',
              '3<-2': 'nor_conv_3x3'}
    fidelity = {'epoch': 199}

    result = b.objective_function(configuration=config, fidelity=fidelity, data_seed=(777, 888, 999))
    assert result is not None
    assert result['function_value'] == pytest.approx(55.2167, abs=0.1)
    assert result['cost'] == pytest.approx(57119.22, abs=0.1)
    assert result['info']['valid_precision'] == result['function_value']
    assert result['info']['valid_cost'] == result['cost']


def test_nasbench201_fidelity_space():
    fs = LocalCifar10ValidNasBench201MOBenchmark.get_fidelity_space()
    assert len(fs.get_hyperparameters()) == 1


def test_nasbench201_config():

    cs = LocalCifar10ValidNasBench201MOBenchmark.get_configuration_space(seed=0)
    c = cs.sample_configuration()

    func = LocalCifar10ValidNasBench201MOBenchmark.config_to_structure_func(4)
    struct = func(c)
    assert struct.__repr__() == '_Structure(4 nodes with |nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+' \
                                '|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_3x3~2|)'
    assert len(struct) == 4
    assert struct[0] == (('nor_conv_1x1', 0),)

    struct_str = struct.tostr()
    assert struct_str == '|nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+' \
                         '|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_3x3~2|'
