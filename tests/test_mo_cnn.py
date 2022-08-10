import pytest


def test_mo_cnn_seeding():
    from hpobench.container.benchmarks.mo.cnn_benchmark import FlowerCNNBenchmark
    b1 = FlowerCNNBenchmark(rng=0)
    b2 = FlowerCNNBenchmark(rng=0)
    test_config = {
        'batch_norm': True, 'batch_size': 71, 'conv_layer_0': 194,  'conv_layer_1': 152,
        'conv_layer_2': 92, 'fc_layer_0': 65, 'fc_layer_1': 19, 'fc_layer_2': 273,
        'global_avg_pooling': True, 'kernel_size': 5, 'learning_rate_init': 0.09091283280651452,
        'n_conv_layers': 2, 'n_fc_layers': 2
    }

    result_1 = b1.objective_function(test_config, rng=1, fidelity={'budget': 3})
    result_2 = b2.objective_function(test_config, rng=1, fidelity={'budget': 3})
    for metric in result_1['function_value'].keys():
        assert result_1['function_value'][metric] == pytest.approx(result_2['function_value'][metric], abs=0.001)


def test_mo_cnn_benchmark():
    from hpobench.container.benchmarks.mo.cnn_benchmark import FlowerCNNBenchmark

    # Check Seeding
    benchmark = FlowerCNNBenchmark(rng=0)
    cs = benchmark.get_configuration_space(seed=0)
    cfg_1 = cs.sample_configuration()

    cs = benchmark.get_configuration_space(seed=0)
    cfg_2 = cs.sample_configuration()

    assert cfg_1 == cfg_2

    test_config = {
        'batch_norm': True, 'batch_size': 71, 'conv_layer_0': 194,  'conv_layer_1': 152,
        'conv_layer_2': 92, 'fc_layer_0': 65, 'fc_layer_1': 19, 'fc_layer_2': 273,
        'global_avg_pooling': True, 'kernel_size': 5, 'learning_rate_init': 0.09091283280651452,
        'n_conv_layers': 2, 'n_fc_layers': 2
    }

    result_1 = benchmark.objective_function(test_config, rng=1, fidelity={'budget': 3})
    result_2 = benchmark.objective_function(test_config, rng=1, fidelity={'budget': 3})
    print(f'MO CNN: Valid Accuracy = {result_1["info"]["valid_accuracy"]}')
    print(f'MO CNN: Train Accuracy = {result_1["info"]["train_accuracy"]}')
    # assert result_1['info']['train_accuracy'] == pytest.approx(0.1044, rel=0.001)
    # assert result_1['info']['valid_accuracy'] == pytest.approx(0.1029, rel=0.001)
    assert result_1['info']['valid_accuracy'] == pytest.approx(1 - result_1['function_value']['negative_accuracy'], abs=0.001)
    assert result_1['info']['train_accuracy'] == result_2['info']['train_accuracy']
