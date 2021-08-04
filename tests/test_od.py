import pytest


def test_ocsvm():
    from hpobench.container.benchmarks.od.od_benchmarks import ODOneClassSupportVectorMachine
    seed = 6
    benchmark = ODOneClassSupportVectorMachine("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    result = benchmark.objective_function_test(configuration=config, rng=seed)
    print(config['gamma'], result['function_value'])
    assert config['gamma'] == pytest.approx(0.065674, abs=0.0001)
    assert result['function_value'] == pytest.approx(0.08180, abs=0.001)


def test_kde():
    from hpobench.container.benchmarks.od.od_benchmarks import  ODKernelDensityEstimation
    seed = 6
    benchmark = ODKernelDensityEstimation("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    result = benchmark.objective_function_test(configuration=config, rng=seed)
    print(config['kernel'], config['bandwidth'], result['function_value'])

    assert config['kernel'] == "exponential"
    assert config['bandwidth'] == pytest.approx(15.2274, abs=0.001)
    assert result['function_value'] == pytest.approx(0.14409, abs=0.0001)


def test_ae():
    from hpobench.container.benchmarks.od.od_benchmarks import ODAutoencoder
    seed = 6
    benchmark = ODAutoencoder("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    result = benchmark.objective_function(configuration=config, rng=seed)
    print(config['dropout_rate'], result['function_value'])

    assert config['dropout_rate'] == pytest.approx(0.69512, abs=0.00001)
    assert result['function_value'] == pytest.approx(0.2833, abs=0.0001)
