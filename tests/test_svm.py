import pytest
from hpobench.container.benchmarks.ml.svm_benchmark import SVMBenchmark
import logging


logging.basicConfig(level=logging.DEBUG)

task_ids = [
    10101,53,146818,146821,9952,146822,31,3917,168912,3,167119,12,146212,168911,
    9981,168329,167120,14965,146606,168330
]


def test_svm_init():
    task_id = 146818
    assert task_id in task_ids
    benchmark = SVMBenchmark(task_id=task_id)

    fs = benchmark.get_fidelity_space(seed=0)
    fidelity = fs.sample_configuration().get_dictionary()
    assert fidelity['subsample'] == pytest.approx(0.59393, abs=0.001)

    meta = benchmark.get_meta_information()
    assert meta is not None

    cs = benchmark.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()
    assert config['C'] == pytest.approx(1.9673, abs=0.001)
    assert config['gamma'] == pytest.approx(19.7501, abs=0.001)

    result = benchmark.objective_function(configuration=config, fidelity=fidelity)
    assert result['function_value'] == pytest.approx(0.4439, abs=0.1)
    assert result['cost'] is not None

    result = benchmark.objective_function_test(configuration=config, fidelity=fidelity)
    assert result['function_value'] == pytest.approx(0.4493, abs=0.1)
    assert result['cost'] is not None

    result = benchmark.objective_function_test(configuration=config)
    assert result['function_value'] == pytest.approx(0.4493, abs=0.1)
    assert result['cost'] is not None
