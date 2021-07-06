import pytest
import logging
logging.basicConfig(level=logging.DEBUG)


def test_ocsvm():
    from hpobench.benchmarks.od.ocsvm_benchmark import ODOneClassSupportVectorMachine
    seed=6
    benchmark = ODOneClassSupportVectorMachine("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    assert config['gamma'] == pytest.approx(0.065674, abs=0.0001)

    result = benchmark.objective_function_test(configuration=config, rng=seed)
    assert result['function_value'] == pytest.approx(0.08180, abs=0.001)


def test_kde():
    from hpobench.benchmarks.od.kde_benchmark import ODKernelDensityEstimation
    seed=6
    benchmark = ODKernelDensityEstimation("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    assert config['kernel'] == "exponential"
    assert config['bandwidth'] == pytest.approx(15.2274, abs=0.001)

    result = benchmark.objective_function_test(configuration=config, rng=seed)
    assert result['function_value'] == pytest.approx(0.14409, abs=0.0001)


def test_ae():
    from hpobench.benchmarks.od.ae_benchmark import ODAutoencoder
    seed=6
    benchmark = ODAutoencoder("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    assert config['dropout_rate'] == pytest.approx(0.07957, abs=0.00001)

    result = benchmark.objective_function(configuration=config, rng=seed)
    assert result['function_value'] == pytest.approx(0.3008, abs=0.0001)
