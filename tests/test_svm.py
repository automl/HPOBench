import pytest

from hpobench.container.benchmarks.ml.svm_benchmark_old import SupportVectorMachine
from hpobench.util.openml_data_manager import get_openmlcc18_taskids

task_ids = get_openmlcc18_taskids()

import logging
logging.basicConfig(level=logging.DEBUG)


def test_svm_init():
    benchmark = SupportVectorMachine(task_id=task_ids[0])

    fs = benchmark.get_fidelity_space(seed=0)
    fidelity = fs.sample_configuration().get_dictionary()
    assert fidelity['dataset_fraction'] == pytest.approx(0.54881, abs=0.001)

    meta = benchmark.get_meta_information()
    assert meta is not None

    cs = benchmark.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()
    assert config['C'] == pytest.approx(0.9762, abs=0.001)
    assert config['gamma'] == pytest.approx(4.3037, abs=0.001)

    result = benchmark.objective_function(configuration=config, fidelity=fidelity)
    assert result['function_value'] == pytest.approx(0.4837, abs=0.1)
    assert result['cost'] is not None

    with pytest.raises(AssertionError):
        result = benchmark.objective_function_test(configuration=config, fidelity=fidelity)

    result = benchmark.objective_function_test(configuration=config)
    assert result['function_value'] == pytest.approx(0.4648, abs=0.1)
    assert result['cost'] is not None
