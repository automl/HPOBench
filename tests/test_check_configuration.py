import unittest
from typing import Dict, Union

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from hpolib.abstract_benchmark import AbstractBenchmark


class TestCheckUnittest(unittest.TestCase):

    def setUp(self):
        class Dummy():
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

        self.foo = Dummy()

    def test_config_decorator(self):
        @AbstractBenchmark._configuration_as_dict
        def tmp(_, configuration: Dict, **kwargs):
            return configuration

        ret = tmp(self=self.foo, configuration=self.foo.configuration_space.sample_configuration())
        self.assertIsInstance(ret, Dict)

        @AbstractBenchmark._check_configuration
        def tmp(_, configuration: Dict, **kwargs):
            return configuration

        tmp(self=self.foo, configuration={"flt": 0.2, "cat": 1, "itg": 1})
        tmp(self=self.foo, configuration=self.foo.configuration_space.sample_configuration())

        self.assertRaises(Exception, tmp, {"self": self.foo, "configuration": {"flt": 0.2, "cat": 1}})
        self.assertRaises(Exception, tmp, {"self": self.foo, "configuration": {"flt": 10000, "cat": 500000}})
        self.assertRaises(Exception, tmp, {"self": self.foo, "configuration": [0.2, 1]})

    def test_fidel_decorator(self):
        @AbstractBenchmark._check_fidelity
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
        class Dummy():
            configuration_space = ConfigurationSpace(seed=1)
            hp1 = UniformFloatHyperparameter("hp1", lower=0.0, upper=0.5, default_value=0.5)
            hp2 = UniformFloatHyperparameter("hp2", lower=1.0, upper=1.5, default_value=1.5)
            hp3 = UniformFloatHyperparameter("hp3", lower=2.0, upper=2.5, default_value=2.5)
            configuration_space.add_hyperparameters([hp1, hp2, hp3])

        self.foo = Dummy()

    def test_config_decorator(self):
        @AbstractBenchmark._check_configuration
        def tmp(_, configuration: Union[Dict, np.ndarray], **kwargs):
            return configuration

        tmp(self=self.foo, configuration=np.array([0.25, 1.25, 2.25]))

        @AbstractBenchmark._configuration_as_array
        def tmp(_, configuration: Dict, **kwargs):
            return configuration

        result = tmp(self=self.foo, configuration=self.foo.configuration_space.get_default_configuration())
        assert np.array_equal(result, np.array([0.5, 1.5, 2.5]))

        result = tmp(self=self.foo, configuration=np.array([0.5, 1.5, 2.5]))
        assert np.array_equal(result, np.array([0.5, 1.5, 2.5]))
