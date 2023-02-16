import pytest

from hpobench.util.container_utils import disable_container_debug, enable_container_debug
from hpobench.util.test_utils import DEFAULT_SKIP_MSG, check_run_all_tests
from hpobench.container.benchmarks.nas.jahs_benchmarks import (
    JAHSMOCifar10SurrogateBenchmark, JAHSMOCifar10TabularBenchmark,
)

@pytest.fixture(scope='module')
def enable_debug():
    enable_container_debug()
    yield
    disable_container_debug()


@pytest.mark.skipif(not check_run_all_tests(), reason=DEFAULT_SKIP_MSG)
def test_jahs_mo_surrogate_benchmark(enable_debug):

    b = JAHSMOCifar10SurrogateBenchmark(rng=0,container_source='/home/pm/Dokumente/Code/Data_HPOBench/Container/jahs_benchmark_0.0.1')

    cs_1 = b.get_configuration_space(seed=0)
    config_1 = cs_1.sample_configuration()
    cs_2 = b.get_configuration_space(seed=0)
    config_2 = cs_2.sample_configuration()
    assert config_1 == config_2

    config = {
        'Activation': 'ReLU',
        'LearningRate': 0.06005288202683084,
        'N': 1,
        'Op1': 3,
        'Op2': 3,
        'Op3': 3,
        'Op4': 1,
        'Op5': 3,
        'Op6': 2,
        'Optimizer': 'SGD',
        'Resolution': 0.25,
        'TrivialAugment': False,
        'W': 16,
        'WeightDecay': 1.4795816345422874e-05,
        }
    result = b.objective_function(configuration=config)
    assert result['function_value']['valid-misclassification_rate'] == pytest.approx(40.82594680786133, abs=0.1)
    assert result['cost'] == pytest.approx(6812.357421875, abs=0.1)
    assert result['info']['valid-misclassification_rate'] == result['function_value']['valid-misclassification_rate']

    result_2 = b.objective_function(configuration=config, fidelity={'nepochs': 200})
    assert result['function_value']['valid-misclassification_rate'] \
           == result_2['function_value']['valid-misclassification_rate']

    with pytest.raises(ValueError):
        b.objective_function_test(configuration=config, fidelity={'nepochs': 201})
