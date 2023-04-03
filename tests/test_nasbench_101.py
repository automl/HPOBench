import pytest
import numpy as np

from hpobench.container.benchmarks.nas.nasbench_101 import (
    NASCifar10ABenchmark, NASCifar10BBenchmark, NASCifar10CBenchmark,
    NASCifar10AMOBenchmark, NASCifar10BMOBenchmark, NASCifar10CMOBenchmark,
)

from hpobench.util.container_utils import disable_container_debug, enable_container_debug
from hpobench.util.test_utils import DEFAULT_SKIP_MSG, check_run_all_tests

# from hpobench.util.test_utils import enable_all_tests
# enable_all_tests()


@pytest.fixture(scope='module')
def enable_debug():
    enable_container_debug()
    yield
    disable_container_debug()


@pytest.mark.skipif(not check_run_all_tests(), reason=DEFAULT_SKIP_MSG)
def test_nasbench101_A_SO(enable_debug):

    b = NASCifar10ABenchmark(rng=0)
    cs_1 = b.get_configuration_space(seed=0)
    config_1 = cs_1.sample_configuration()
    cs_2 = b.get_configuration_space(seed=0)
    config_2 = cs_2.sample_configuration()
    assert config_1 == config_2

    assert len(b.get_fidelity_space()) == 1

    config = {
        'edge_0': 0, 'edge_1': 0, 'edge_10': 0, 'edge_11': 1, 'edge_12': 1, 'edge_13': 0, 'edge_14': 1, 'edge_15': 0,
        'edge_16': 0, 'edge_17': 1, 'edge_18': 1, 'edge_19': 0, 'edge_2': 0, 'edge_20': 1, 'edge_3': 0, 'edge_4': 0,
        'edge_5': 1, 'edge_6': 1, 'edge_7': 0, 'edge_8': 0, 'edge_9': 0, 'op_node_0': 'maxpool3x3',
        'op_node_1': 'conv1x1-bn-relu', 'op_node_2': 'conv3x3-bn-relu', 'op_node_3': 'conv3x3-bn-relu',
        'op_node_4': 'conv3x3-bn-relu'
    }

    result = b.objective_function(configuration=config, fidelity={'budget': 108}, run_index=(0, 1, 2))
    assert result['function_value'] == pytest.approx(0.1659655372301737, abs=0.1)
    assert result['cost'] == pytest.approx(853.5010070800781, abs=0.1)
    assert 1 - np.mean(result['info']['valid_accuracies']) == result['function_value']

    with pytest.raises(AssertionError):
        result = b.objective_function_test(configuration=config, fidelity={'epoch': 109})


@pytest.mark.skipif(not check_run_all_tests(), reason=DEFAULT_SKIP_MSG)
def test_nasbench101_C_MO(enable_debug):
    b = NASCifar10CMOBenchmark(rng=0)
    cs_1 = b.get_configuration_space(seed=0)
    config_1 = cs_1.sample_configuration()
    cs_2 = b.get_configuration_space(seed=0)
    config_2 = cs_2.sample_configuration()
    assert config_1 == config_2

    assert len(b.get_fidelity_space()) == 1

    config = {
        'edge_0': 0.9446689170495839, 'edge_1': 0.1289262976548533, 'edge_10': 0.09710127579306127,
        'edge_11': 0.09394051075844168, 'edge_12': 0.5722519057908734, 'edge_13': 0.30157481667454933,
        'edge_14': 0.9194826137446735, 'edge_15': 0.3599780644783639, 'edge_16': 0.589909976354571,
        'edge_17': 0.4536968445560453, 'edge_18': 0.21550767711355845, 'edge_19': 0.18327983621407862,
        'edge_2': 0.5864101661863267, 'edge_20': 0.47837030703998806, 'edge_3': 0.05342718178682526,
        'edge_4': 0.6956254456388572, 'edge_5': 0.3068100995451961, 'edge_6': 0.399025321703102,
        'edge_7': 0.15941446344895593, 'edge_8': 0.23274412927905685, 'edge_9': 0.0653042071517802, 'num_edges': 9,
        'op_node_0': 'conv1x1-bn-relu', 'op_node_1': 'maxpool3x3', 'op_node_2': 'conv1x1-bn-relu',
        'op_node_3': 'maxpool3x3', 'op_node_4': 'maxpool3x3'
    }

    result = b.objective_function(configuration=config, fidelity={'budget': 108}, run_index=(0, 1, 2))
    assert result['function_value']['misclassification_rate'] == pytest.approx(0.11985842386881507, abs=0.1)
    assert result['function_value']['trainable_parameters'] == 1115277
    assert result['cost'] == pytest.approx(3175.9591064453125, abs=0.1)
    assert 1 - np.mean(result['info']['valid_accuracies']) == result['function_value']['misclassification_rate']

    with pytest.raises(AssertionError):
        result = b.objective_function_test(configuration=config, fidelity={'epoch': 109})
