import logging
import pytest

logging.basicConfig(level=logging.DEBUG)


def test_wikitext_benchmark():
    from hpobench.benchmarks.mo.lm_benchmark import LanguageModelBenchmark

    # Check Seeding
    benchmark = LanguageModelBenchmark(rng=0)
    cs = benchmark.get_configuration_space(seed=1)
    cfg_1 = cs.sample_configuration()

    cs = benchmark.get_configuration_space(seed=1)
    cfg_2 = cs.sample_configuration()

    assert cfg_1 == cfg_2

    test_config = {
        'batch_size': 144, 'clip': 1.458859796107597, 'dropout': 0.5967357423109274,
        'emsize': 575, 'lr': 5.245378070737081, 'lr_factor': 15
    }

    result_1 = benchmark.objective_function(test_config, rng=1, fidelity={'budget': 1})
    result_2 = benchmark.objective_function(test_config, rng=1, fidelity={'budget': 1})
    assert result_1['info']['train_accuracy'] == pytest.approx(0.76145, rel=0.001)
    assert result_1['info']['train_accuracy'] == result_2['info']['train_accuracy']
