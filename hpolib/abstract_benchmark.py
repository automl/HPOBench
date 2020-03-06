""" Base-class of all benchmarks """

import abc
from typing import Tuple, Union, Dict, List

import ConfigSpace
import numpy as np

from hpolib.util import rng_helper


class AbstractBenchmark(object, metaclass=abc.ABCMeta):

    def __init__(self, rng: Union[int, np.random.RandomState, None] = None):
        """
        Interface for benchmarks.

        A benchmark consists of two building blocks, the target function and
        the configuration space. Furthermore it can contain additional
        benchmark-specific information such as the location and the function
        value of the global optima.
        New benchmarks should be derived from this base class or one of its
        child classes.

        Parameters
        ----------
        rng: int, np.random.RandomState, None
            The default random state for the benchmark. If type is int, a
            np.random.RandomState with seed `rng` is created. If type is None,
            create a new random state.
        """

        self.rng = rng_helper.get_rng(rng=rng)
        self.configuration_space = self.get_configuration_space()

    @abc.abstractmethod
    def objective_function(self, configuration: Dict, *args, **kwargs) -> dict:
        """
        Objective function.

        Override this function to provide your benchmark function. This
        function will be called by one of the evaluate functions. For
        flexibility you have to return a dictionary with the only mandatory
        key being `function_value`, the objective function value for the
        `configuration` which was passed. By convention, all benchmarks are
        minimization problems.

        Note
        ----
        It might be useful to pass a `seed` argument to the function call to
        bypass the default "seed" generator. Only using the default random
        state (`self.rng`) could lead to an overfitting towards the
        `self.rng`'s seed.

        For an example, how to pass a random state to the objective function
        see
        :py:func:`~hpolib.benchmarks.ml.xgboost_benchmark.XGBoostBenchmark.objective_function`.

        Parameters
        ----------
        configuration : Dict

        Returns
        -------
        Dict
            Must contain at least the key `function_value`.
        """
        pass

    @abc.abstractmethod
    def objective_function_test(self, configuration: Dict, *args, **kwargs) -> Dict:
        """
        If there is a different objective function for offline testing, e.g
        testing a machine learning on a hold extra test set instead
        on a validation set override this function here.

        Parameters
        ----------
        configuration : Dict

        Returns
        -------
        Dict
            Must contain at least the key `function_value`.
        """
        pass

    @staticmethod
    def _check_configuration(foo):
        """
        Decorator to enable checking the input configuration.

        Uses the check_configuration of the ConfigSpace class to ensure
        that all specified values are valid, and no conditionals are violated

        Can be combined with the _configuration_as_array decorator.
        """
        def wrapper(self, configuration, **kwargs):
            if not isinstance(configuration, ConfigSpace.Configuration):
                try:
                    config_dict = {k: configuration[i] for (i, k) in enumerate(self.configuration_space)}
                    config = ConfigSpace.Configuration(self.configuration_space, config_dict)
                except Exception as e:
                    raise Exception('Error during the conversion of the provided into a ConfigSpace.Configuration'
                                    ' object') from e
            else:
                config = configuration
            self.configuration_space.check_configuration(config)
            return foo(self, configuration, **kwargs)
        return wrapper

    @staticmethod
    def _configuration_as_array(foo, data_type=np.float):
        """
        Decorator to allow the first input argument to 'objective_function' to
        be an array.

        For all continuous benchmarks it is often required that the input to
        the benchmark can be a (NumPy) array. By adding this to the objective
        function, both inputs types, ConfigSpace.Configuration and array,
        are possible.

        Can be combined with the _check_configuration decorator.
        """
        def wrapper(self, configuration, **kwargs):
            if isinstance(configuration, ConfigSpace.Configuration):
                config_array = np.array([configuration[k] for k in configuration], dtype=data_type)
            else:
                config_array = configuration
            return foo(self, config_array, **kwargs)
        return wrapper

    def __call__(self, configuration: Dict, **kwargs) -> float:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    def test(self, n_runs: int = 5, *args, **kwargs) -> Tuple[List, List]:
        """
        Draws some random configuration and call objective_function(_test).

        Parameters
        ----------
        n_runs : int
            number of random configurations to draw and evaluate

        Returns
        -------
        Tuple[List, List]
        """
        train_rvals = []
        test_rvals = []

        for _ in range(n_runs):
            configuration = self.configuration_space.sample_configuration()
            train_rvals.append(self.objective_function(configuration, *args, **kwargs))
            test_rvals.append(self.objective_function_test(configuration, *args, **kwargs))

        return train_rvals, test_rvals

    @staticmethod
    @abc.abstractmethod
    def get_configuration_space() -> ConfigSpace.ConfigurationSpace:
        """ Defines the configuration space for each benchmark.

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's parameters
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_meta_information() -> Dict:
        """ Provides some meta information about the benchmark.

        Returns
        -------
        Dict
            some human-readable information

        """
        raise NotImplementedError()
