import sys
from typing import Dict, List

import pytest

from hpobench.container.benchmarks.surrogates.yahpo_gym import YAHPOGymBenchmark, YAHPOGymMOBenchmark


def test_yahpo_init():
    b = YAHPOGymBenchmark(scenario="lcbench", instance="167152", objective="val_accuracy")

    fs = b.get_fidelity_space(seed=0)
    fidelity = fs.sample_configuration().get_dictionary()
    assert isinstance(fidelity, Dict)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()

    # Some tests are dependent on the python version.
    if sys.version.startswith('3.9'):
        assert fidelity['epoch'] == pytest.approx(29, abs=0.001)
        assert config['OpenML_task_id'] == "167152"
        assert config['num_layers'] == pytest.approx(4, abs=0.001)
        assert config['max_units'] == pytest.approx(289, abs=0.0001)
        assert config['weight_decay'] == pytest.approx(0.04376, abs=0.001)
        assert config['learning_rate'] == pytest.approx(0.01398, abs=0.0001)
        assert config['batch_size'] == pytest.approx(106, abs=0.001)

    constant_fidelity = {'epoch': 29}
    constant_config = {
        'OpenML_task_id': '167152', 'batch_size': 106, 'learning_rate': 0.013981961408994055,
        'max_dropout': 0.6027633760716439, 'max_units': 289, 'momentum': 0.47705277141162516,
        'num_layers': 4, 'weight_decay': 0.04376434525415663
    }

    result = b.objective_function(configuration=constant_config, fidelity=constant_fidelity)
    assert result['function_value'] == pytest.approx(61.297, abs=0.1)
    assert result['cost'] == pytest.approx(119.4965, abs=0.1)
    assert isinstance(result['info'], Dict)


def test_yahpo_mo():
    b = YAHPOGymMOBenchmark(scenario="lcbench", instance="167152")

    fs = b.get_fidelity_space(seed=0)
    fidelity = fs.sample_configuration().get_dictionary()
    assert isinstance(fidelity, Dict)

    cs = b.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()

    # Some tests are dependent on the python version.
    if sys.version.startswith('3.9'):
        assert fidelity['epoch'] == pytest.approx(29, abs=0.001)
        assert config['OpenML_task_id'] == "167152"
        assert config['num_layers'] == pytest.approx(4, abs=0.001)
        assert config['max_units'] == pytest.approx(289, abs=0.0001)
        assert config['weight_decay'] == pytest.approx(0.04376, abs=0.001)
        assert config['learning_rate'] == pytest.approx(0.01398, abs=0.0001)
        assert config['batch_size'] == pytest.approx(106, abs=0.001)

    constant_fidelity = {'epoch': 29}
    constant_config = {
        'OpenML_task_id': '167152', 'batch_size': 106, 'learning_rate': 0.013981961408994055,
        'max_dropout': 0.6027633760716439, 'max_units': 289, 'momentum': 0.47705277141162516,
        'num_layers': 4, 'weight_decay': 0.04376434525415663
    }

    result = b.objective_function(configuration=constant_config, fidelity=constant_fidelity)
    assert isinstance(result['function_value'], Dict)
    assert result['function_value']['val_accuracy'] == pytest.approx(61.2971, abs=0.0001)
    assert result['cost'] == pytest.approx(119.4965, abs=0.0001)

    names = b.get_objective_names()
    assert isinstance(names, List)
    assert len(names) == 6
    assert names[2] == 'val_cross_entropy'
