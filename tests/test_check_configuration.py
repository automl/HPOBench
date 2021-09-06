import unittest
from typing import Dict, Union

import numpy as np
import pytest

from ConfigSpace import ConfigurationSpace, Configuration, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, NotEqualsCondition

from hpobench.abstract_benchmark import AbstractBenchmark


class TestCheckUnittest(unittest.TestCase):

    def setUp(self):
        class Dummy:
            configuration_space = ConfigurationSpace(seed=1)
            flt = UniformFloatHyperparameter("flt", lower=0.0, upper=1.0)
            cat = CategoricalHyperparameter("cat", choices=(1, "a"))
            itg = UniformIntegerHyperparameter("itg", lower=0, upper=10)
            configuration_space.add_hyperparameters([flt, cat, itg])

            fidelity_space = ConfigurationSpace(seed=1)
            f1 = UniformFloatHyperparameter("f_flt", lower=0.0, upper=1.0)
            f2 = CategoricalHyperparameter("f_cat", choices=(1, "a"))
            f3 = UniformIntegerHyperparameter("f_itg", lower=0, upper=10)
            fidelity_space.add_hyperparameters([f1, f2, f3])

            def get_fidelity_space(self):
                return self.fidelity_space

            _check_and_cast_configuration = AbstractBenchmark._check_and_cast_configuration
            _check_and_cast_fidelity = AbstractBenchmark._check_and_cast_fidelity

        self.foo = Dummy()

    def test_config_decorator(self):
        @AbstractBenchmark.check_parameters
        def tmp(_, configuration: Dict, fidelity: Dict, **kwargs):
            return configuration

        ret = tmp(self=self.foo, configuration=self.foo.configuration_space.sample_configuration())
        self.assertIsInstance(ret, Dict)

        @AbstractBenchmark.check_parameters
        def tmp(_, configuration: Dict, fidelity: Dict, **kwargs):
            return configuration

        tmp(self=self.foo, configuration={"flt": 0.2, "cat": 1, "itg": 1})
        tmp(self=self.foo, configuration=self.foo.configuration_space.sample_configuration())

        self.assertRaises(Exception, tmp, {"self": self.foo, "configuration": {"flt": 0.2, "cat": 1}})
        self.assertRaises(Exception, tmp, {"self": self.foo, "configuration": {"flt": 10000, "cat": 500000}})
        self.assertRaises(Exception, tmp, {"self": self.foo, "configuration": [0.2, 1]})

    def test_fidel_decorator(self):
        @AbstractBenchmark.check_parameters
        def tmp(_, configuration: Dict, fidelity: Dict, **kwargs):
            return configuration, fidelity, kwargs

        sample_fidel = dict(self.foo.get_fidelity_space().get_default_configuration())

        _, ret, _ = tmp(self=self.foo,
                        configuration=self.foo.configuration_space.sample_configuration(),
                        fidelity=sample_fidel)
        self.assertEqual(ret, sample_fidel)

        less_fidel = {"f_cat": 1}
        _, ret, _ = tmp(self=self.foo,
                        configuration=self.foo.configuration_space.sample_configuration(),
                        fidelity=less_fidel)
        self.assertEqual(ret, sample_fidel)

        _, ret, _ = tmp(self=self.foo,
                        configuration=self.foo.configuration_space.sample_configuration())
        self.assertEqual(ret, sample_fidel)

        with pytest.raises(ValueError):
            tmp(self=self.foo, configuration=self.foo.configuration_space.sample_configuration(),
                f_cat=1)

        self.assertRaises(Exception, tmp, {"self": self.foo,
                                           "configuration": self.foo.configuration_space.sample_configuration(),
                                           "fidelity": {"f_cat": "b"}})
        self.assertRaises(TypeError, tmp, {"self": self.foo,
                                           "configuration": self.foo.configuration_space.sample_configuration(),
                                           "fidelity": [0.1]})

class TestCheckUnittest2(unittest.TestCase):

    def setUp(self):
        from hpobench.abstract_benchmark import AbstractBenchmark
        class Dummy():
            configuration_space = ConfigurationSpace(seed=1)
            hp1 = UniformFloatHyperparameter("hp1", lower=0.0, upper=0.5, default_value=0.5)
            hp2 = UniformFloatHyperparameter("hp2", lower=1.0, upper=1.5, default_value=1.5)
            hp3 = UniformFloatHyperparameter("hp3", lower=2.0, upper=2.5, default_value=2.5)
            configuration_space.add_hyperparameters([hp1, hp2, hp3])

            _check_and_cast_configuration = AbstractBenchmark._check_and_cast_configuration
            _check_and_cast_fidelity = AbstractBenchmark._check_and_cast_fidelity

            fidelity_space = ConfigurationSpace(seed=1)
            fidelity_space.add_hyperparameter(UniformFloatHyperparameter('fidelity1', lower=0., upper=1., default_value=1.))
        self.foo = Dummy()

    def test_config_decorator(self):
        @AbstractBenchmark.check_parameters
        def tmp(_, configuration: Union[Dict, np.ndarray], fidelity: Dict, **kwargs):
            return configuration, fidelity

        hps = dict(hp1=0.25, hp2=1.25, hp3=2.25)
        configuration = Configuration(self.foo.configuration_space, hps)
        config, fidel = tmp(self=self.foo, configuration=configuration, fidelity=None)

        assert isinstance(config, Dict)
        assert isinstance(fidel, Dict)
        assert fidel['fidelity1'] == 1.0


def test_remove_inactive_parameter():
    configuration_space = ConfigurationSpace(seed=1)
    hp1 = CategoricalHyperparameter("hp1", choices=[0, 1])
    hp2 = CategoricalHyperparameter("hp2", choices=['a'])
    hp3 = UniformIntegerHyperparameter("hp3", lower=0, upper=5, default_value=5)
    configuration_space.add_hyperparameters([hp1, hp2, hp3])

    # If hp1 = 0, then don't allow hp2
    not_condition = NotEqualsCondition(hp2, hp1, 0)
    configuration_space.add_condition(not_condition)

    allowed_cfg = Configuration(configuration_space, {'hp1': 1, 'hp2': 'a', 'hp3': 5})
    not_allowed = {'hp1': 0, 'hp2': 'a', 'hp3': 5}

    with pytest.raises(ValueError):
        Configuration(configuration_space, not_allowed)

    # No inactive hp - case: config is CS.configuration
    transformed = AbstractBenchmark._check_and_cast_configuration(allowed_cfg, configuration_space)
    assert transformed.get_dictionary() == {'hp1': 1, 'hp2': 'a', 'hp3': 5}

    # No inactive hp - case: config is dict
    transformed = AbstractBenchmark._check_and_cast_configuration(allowed_cfg.get_dictionary(), configuration_space)
    assert transformed.get_dictionary() == {'hp1': 1, 'hp2': 'a', 'hp3': 5}

    # Remove inactive: - case: config is CS.configuration
    not_allowed_cs = Configuration(configuration_space, {'hp1': 0, 'hp2': 'a', 'hp3': 5},
                                   allow_inactive_with_values=True)
    transformed = AbstractBenchmark._check_and_cast_configuration(not_allowed_cs, configuration_space)
    assert transformed.get_dictionary() == {'hp1': 0, 'hp3': 5}

    # Remove inactive: - case: config is dict
    transformed = AbstractBenchmark._check_and_cast_configuration(not_allowed, configuration_space)
    assert transformed.get_dictionary() == {'hp1': 0, 'hp3': 5}
