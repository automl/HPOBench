import pytest
import typing

from hpobench.container.benchmarks.surrogates.yahpo_gym import YAHPOGymBenchmark

import logging
logging.basicConfig(level=logging.DEBUG)


def test_yahpo_init():
    benchmark = YAHPOGymBenchmark(scenario="lcbench",  instance = "167152", objective = "val_accuracy")

    fs = b.get_fidelity_space(seed = 0)
    fidelity = fs.sample_configuration().get_dictionary()
    assert fidelity['epoch'] == pytest.approx(29, abs=0.001)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()
    assert config['OpenML_task_id'] == "167152"
    assert config['num_layers'] == pytest.approx(4, abs=0.001)
    assert config['max_units'] == pytest.approx(289, abs=0.0001)
    assert config['weight_decay'] == pytest.approx(0.04376, abs=0.001)
    assert config['learning_rate'] == pytest.approx(0.01398, abs=0.0001)
    assert config['batch_size'] == pytest.approx(106, abs=0.001)

    result = b.objective_function(configuration=config, fidelity=fidelity)
    assert result['function_value'] == pytest.approx(61.297, abs=0.1)
    assert result['cost'] == pytest.approx(119.4965, abs = 0.1)
    assert isinstance(result['info'], typing.Dict)
    assert [k for k in result['info']['objectives'].keys()] == b.config.y_names

