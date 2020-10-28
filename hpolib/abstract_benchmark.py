""" Base-class of all benchmarks """

import abc
from typing import Union, Dict, List
import functools

import logging
import ConfigSpace
import numpy as np

from hpolib.util import rng_helper

logger = logging.getLogger('AbstractBenchmark')


class AbstractBenchmark(abc.ABC, metaclass=abc.ABCMeta):

    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, **kwargs):
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
        self.fidelity_space = self.get_fidelity_space()

    @abc.abstractmethod
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Objective function.

        Override this function to provide your benchmark function. This
        function will be called by one of the evaluate functions. For
        flexibility you have to return a dictionary with the only mandatory
        key being `function_value`, the objective function value for the
        `configuration` which was passed. By convention, all benchmarks are
        minimization problems.

        Parameters
        ----------
        configuration : Dict
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            It might be useful to pass a `rng` argument to the function call to
            bypass the default "seed" generator. Only using the default random
            state (`self.rng`) could lead to an overfitting towards the
            `self.rng`'s seed.

        Returns
        -------
        Dict
            Must contain at least the key `function_value` and `cost`.
        """
        NotImplementedError()

    @abc.abstractmethod
    def objective_function_test(self, configuration: Union[ConfigSpace.Configuration, Dict],
                                fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        If there is a different objective function for offline testing, e.g
        testing a machine learning on a hold extra test set instead
        on a validation set override this function here.

        Parameters
        ----------
        configuration : Dict
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            see :py:func:`~HPOlib2.abstract_benchmark.objective_function`

        Returns
        -------
        Dict
            Must contain at least the key `function_value` and `cost`.
        """
        NotImplementedError()

    @staticmethod
    def _check_configuration(foo):
        """
        Decorator to enable checking the input configuration and, if given, fidelity parameters.

        Uses the check_configuration of the ConfigSpace class to ensure
        that all specified values are valid, and no conditionals are violated

        Can be combined with the _configuration_as_array decorator.
        """

        # Copy all documentation from the underlying function except the annotations.
        @functools.wraps(wrapped=foo, assigned=('__module__', '__name__', '__qualname__', '__doc__',))
        def wrapper(self, configuration: Union[np.ndarray, List, ConfigSpace.Configuration, Dict], **kwargs):

            try:
                if isinstance(configuration, (np.ndarray, List)):
                    config_dict = {k: configuration[i] for (i, k) in enumerate(self.configuration_space)}
                    config = ConfigSpace.Configuration(self.configuration_space, config_dict)
                elif isinstance(configuration, dict):
                    config = ConfigSpace.Configuration(self.configuration_space, configuration)
                elif isinstance(configuration, ConfigSpace.Configuration):
                    config = configuration
                else:
                    config = None
            except Exception as e:
                logger.error('Error during the conversion of the provided configuration '
                             'into a ConfigSpace.Configuration object')
                raise e

            if config is None:
                raise TypeError(f'Configuration has to be from type np.ndarray, dict, or ConfigSpace.Configuration but '
                                f'was {type(configuration)}')

            self.configuration_space.check_configuration(config)
            return foo(self, configuration, **kwargs)
        return wrapper

    @staticmethod
    def _check_fidelity(foo):
        """
        Decorator to enable checking the input fidelity parameters, if any. Wrapped functions are expected to contain
        an optional 'fidelity':keyword argument, which would in turn contain a dictionary of the requested fidelity
        parameters. If any specific parameter is missing or the entire argument is missing, the corresponding default
        values are filled in.

        Uses the check_configuration of the ConfigSpace class to ensure that all specified values are valid, and no
        conditionals are violated.

        Order independent from the _check_configuration decorator, but it does forward all fidelity parameters,
        regardless of input, as a dictionary in the 'fidelity' keyword argument.
        """

        # Copy all documentation from the underlying function except the annotations.
        @functools.wraps(wrapped=foo, assigned=('__module__', '__name__', '__qualname__', '__doc__',))
        def wrapper(self, configuration: Union[np.ndarray, ConfigSpace.Configuration, Dict],
                    fidelity: Union[Dict, ConfigSpace.Configuration, None] = None, **kwargs):

            # Sanity check that there are no fidelities in **kwargs
            for f in self.fidelity_space.get_hyperparameters():
                if f.name in kwargs:
                    raise ValueError(f'Fidelity parameter {f.name} should not be part of kwargs\n'
                                     f'Fidelity: {fidelity}\n Kwargs: {kwargs}')

            # If kwargs contains the 'fidelity' arg, extract any fidelity parameters it contains and fill in
            # default values for the rest.
            default_fidelities = self.fidelity_space.get_default_configuration()
            try:
                if fidelity is None:
                    fidelity = default_fidelities
                if isinstance(fidelity, dict):
                    default_fidelities_cfg = default_fidelities.get_dictionary()
                    fidelity = {k: fidelity.get(k, v) for k, v in default_fidelities_cfg.items()}
                    fidelity = ConfigSpace.Configuration(self.fidelity_space, fidelity)
                elif isinstance(fidelity, ConfigSpace.Configuration):
                    pass
                else:
                    fidelity = None
            except Exception as e:
                logger.error('Error during the conversion of the provided fidelities '
                             'into a FidelitySpace (ConfigSpace.Configuration) object')
                raise e

            if fidelity is None:
                raise TypeError(f'Fidelity has to be an instance of type np.ndarray, dict, or '
                                f'ConfigSpace.Configuration but was {type(configuration)}')

            # Ensure that the extracted fidelity values play well with the defined fidelity space
            self.fidelity_space.check_configuration(fidelity)

            # All benchmarks should work on dictionaries. Cast the fidelity space object to a dictionary.
            fidelity = fidelity.get_dictionary()

            return foo(self, configuration, fidelity=fidelity, **kwargs)
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

    @staticmethod
    def _configuration_as_dict(foo):
        """
        Decorator to cast the ConfigSpace.configuration to a dictionary. This allows the first argument of
        objective_function and objective_function_test to be a ConfigSpace.configuration.

        Can be combined with the _check_configuration decorator.
        """
        def wrapper(self, configuration, **kwargs):
            if isinstance(configuration, ConfigSpace.Configuration):
                configuration = configuration.get_dictionary()
            return foo(self, configuration, **kwargs)
        return wrapper

    def __call__(self, configuration: Dict, **kwargs) -> float:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    @staticmethod
    @abc.abstractmethod
    def get_configuration_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        """ Defines the configuration space for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the configuration space.

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's parameters
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        """ Defines the available fidelity parameters as a "fidelity space" for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the fidelity space.
        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's fidelity parameters
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
