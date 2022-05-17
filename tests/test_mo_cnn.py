import logging
import pytest

logging.basicConfig(level=logging.DEBUG)


def test_mo_cnn_benchmark():
    from hpobench.benchmarks.mo.cnn_benchmark import FlowerCNNBenchmark

    # Check Seeding
    benchmark = FlowerCNNBenchmark(rng=0)
    cs = benchmark.get_configuration_space(seed=0)
    cfg_1 = cs.sample_configuration()

    cs = benchmark.get_configuration_space(seed=0)
    cfg_2 = cs.sample_configuration()

    assert cfg_1 == cfg_2

    result_1 = benchmark.objective_function(cfg_1, rng=1, fidelity={'budget': 5})
    result_2 = benchmark.objective_function(cfg_1, rng=1, fidelity={'budget': 5})

    assert result_1['info']['train_accuracy'] == pytest.approx(0.08676, rel=0.001)
    assert result_1['info']['train_accuracy'] == result_2['info']['train_accuracy']
