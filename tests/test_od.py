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
    from hpobench.container.benchmarks.od.od_benchmarks import ODKernelDensityEstimation
    seed = 6
    benchmark = ODKernelDensityEstimation("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    assert config is not None

    test_config = {'bandwidth': 15.227439996058147, 'kernel': 'tophat', 'scaler': 'Standard'}
    result = benchmark.objective_function_test(configuration=test_config, rng=seed)
    assert result['function_value'] == pytest.approx(0.8675, abs=0.0001)


def test_ae():
    from hpobench.container.benchmarks.od.od_benchmarks import ODAutoencoder
    seed = 6
    benchmark = ODAutoencoder("cardio", rng=seed)

    config = benchmark.get_configuration_space(seed=seed).sample_configuration()
    assert config is not None

    test_config = {'activation': 'tanh', 'batch_normalization': True,
                   'batch_size': 424, 'beta1': 0.8562127972330622, 'beta2': 0.9107549023256032,
                   'dropout': False, 'lr': 0.0013160410886450579, 'num_latent_units': 5,
                   'num_layers': 1, 'scaler': 'MinMax', 'skip_connection': True,
                   'weight_decay': 0.07358821063486902, 'num_units_layer_1': 16}

    result = benchmark.objective_function(configuration=test_config, rng=seed)
    assert result['function_value'] == pytest.approx(0.81378, abs=0.001)
