import logging
import pytest

logging.basicConfig(level=logging.DEBUG)


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

    assert result_1['info']['train_accuracy'] == pytest.approx(0.1029, rel=0.001)
    assert result_1['info']['train_accuracy'] == result_2['info']['train_accuracy']
