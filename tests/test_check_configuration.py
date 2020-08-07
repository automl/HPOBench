from typing import Union, Tuple, Dict, List
import unittest

import numpy as np

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

        try:
            tmp(self=self.foo, configuration=self.foo.configuration_space.sample_configuration(),
                f_cat=1)
        except ValueError as e:
            self.assertEqual(e.__str__(), "Fidelity parameter f_cat should not be part of kwargs")
