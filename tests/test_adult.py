import logging
import pytest

logging.basicConfig(level=logging.DEBUG)


def test_adult_benchmark():
    from hpobench.container.benchmarks.mo.adult_benchmark import AdultBenchmark

    # Check Seeding
    benchmark = AdultBenchmark(rng=0)
    cs = benchmark.get_configuration_space(seed=0)
    cfg_1 = cs.sample_configuration()

    cs = benchmark.get_configuration_space(seed=0)
    cfg_2 = cs.sample_configuration()

    assert cfg_1 == cfg_2

    test_config = {
        'alpha': 0.00046568046379195655, 'beta_1': 0.14382335124614148, 'beta_2': 0.0010007892350251595,
        'fc_layer_0': 4, 'fc_layer_1': 2, 'fc_layer_2': 2, 'fc_layer_3': 3,'n_fc_layers': 4,
        'learning_rate_init': 0.0005343227125594117,
        'tol': 0.0004134759007834719
    }

    result_1 = benchmark.objective_function(test_config, rng=1, fidelity={'budget': 3})
    result_2 = benchmark.objective_function(test_config, rng=1, fidelity={'budget': 3})

    assert result_1['info']['valid_accuracy'] == pytest.approx(0.7539, rel=0.001)
    assert 1 - result_1['info']['valid_accuracy'] == result_1['function_value']['misclassification_rate']
    assert result_1['info']['train_accuracy'] == pytest.approx(0.76145, rel=0.001)
    assert result_1['info']['train_accuracy'] == result_2['info']['train_accuracy']

    result_1 = benchmark.objective_function_test(test_config, rng=1, fidelity={'budget': 3})
    assert 1 - result_1['function_value']['misclassification_rate'] == pytest.approx(0.76377, rel=0.001)
    assert 1 - result_1['function_value']['misclassification_rate'] == result_1['info']['test_accuracy']
