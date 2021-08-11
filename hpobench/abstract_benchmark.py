""" Base-class of all benchmarks """

import abc
from typing import Union, Dict
import functools

import logging
import ConfigSpace
import numpy as np

from ConfigSpace.util import deactivate_inactive_hyperparameters
from hpobench.util import rng_helper

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
        self.configuration_space = self.get_configuration_space(self.rng.randint(0, 10000))
        self.fidelity_space = self.get_fidelity_space(self.rng.randint(0, 10000))

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
            see :py:func:`~HPOBench.abstract_benchmark.objective_function`

        Returns
        -------
        Dict
            Must contain at least the key `function_value` and `cost`.
        """
        NotImplementedError()

    @staticmethod
    def check_parameters(wrapped_function):
        """
        Wrapper for the objective_function and objective_function_test.
        This function verifies the correctness of the input configuration and the given fidelity.

        It ensures that both, configuration and fidelity, don't contain any wrong parameters or any conditions are
        violated.

        If the argument 'fidelity' is not specified or a single parameter is missing, then the corresponding default
        fidelities are filled in.

        We cast them to a ConfigSpace.Configuration object to ensure that no conditions are violated.
        """

        # Copy all documentation from the underlying function except the annotations.
        @functools.wraps(wrapped=wrapped_function, assigned=('__module__', '__name__', '__qualname__', '__doc__',))
        def wrapper(self, configuration: Union[ConfigSpace.Configuration, Dict],
                    fidelity: Union[Dict, ConfigSpace.Configuration, None] = None, **kwargs):

            configuration = AbstractBenchmark._check_and_cast_configuration(configuration, self.configuration_space)

            # Second, evaluate the given fidelities.
            # Sanity check that there are no fidelities in **kwargs
            fidelity = AbstractBenchmark._check_and_cast_fidelity(fidelity, self.fidelity_space, **kwargs)

            # All benchmarks should work on dictionaries. Cast the both objects to dictionaries.
            return wrapped_function(self, configuration.get_dictionary(), fidelity.get_dictionary(), **kwargs)
        return wrapper

    @staticmethod
    def _check_and_cast_configuration(configuration: Union[Dict, ConfigSpace.Configuration],
                                      configuration_space: ConfigSpace.ConfigurationSpace) \
            -> ConfigSpace.Configuration:
        """ Helper-function to evaluate the given configuration.
            Cast it to a ConfigSpace.Configuration and evaluate if it violates its boundaries.

            Note:
                We remove inactive hyperparameters from the given configuration. Inactive hyperparameters are
                hyperparameters that are not relevant for a configuration, e.g. hyperparameter A is only relevant if
                hyperparameter B=1 and if B!=1 then A is inactive and will be removed from the configuration.
                Since the authors of the benchmark removed those parameters explicitly, they should also handle the
                cases that inactive parameters are not present in the input-configuration.
        """

        if isinstance(configuration, dict):
            configuration = ConfigSpace.Configuration(configuration_space, configuration,
                                                      allow_inactive_with_values=True)
        elif isinstance(configuration, ConfigSpace.Configuration):
            configuration = configuration
        else:
            raise TypeError(f'Configuration has to be from type List, np.ndarray, dict, or '
                            f'ConfigSpace.Configuration but was {type(configuration)}')

        all_hps = set(configuration_space.get_hyperparameter_names())
        active_hps = configuration_space.get_active_hyperparameters(configuration)
        inactive_hps = all_hps - active_hps

        if len(inactive_hps) != 0:
            logger.debug(f'There are inactive {len(inactive_hps)} hyperparameter: {inactive_hps}'
                         'Going to remove them from the configuration.')

        configuration = deactivate_inactive_hyperparameters(configuration, configuration_space)
        configuration_space.check_configuration(configuration)

        return configuration

    @staticmethod
    def _check_and_cast_fidelity(fidelity: Union[dict, ConfigSpace.Configuration, None],
                                 fidelity_space: ConfigSpace.ConfigurationSpace, **kwargs) \
            -> ConfigSpace.Configuration:
        """ Helper-function to evaluate the given fidelity object.
            Similar to the checking and casting from above, we validate the fidelity object. To do so, we cast it to a
            ConfigSpace.Configuration object.
            If the fidelity is not specified (None), then we use the default fidelity of the benchmark.
            If the benchmark is a multi-multi-fidelity benchmark and only a subset of the available fidelities is
            specified, we fill the missing ones with their default values.
        """
        # Make a check, that no fidelities are in the kwargs.
        f_in_kwargs = []
        for f in fidelity_space.get_hyperparameters():
            if f.name in kwargs:
                f_in_kwargs.append(f.name)
        if len(f_in_kwargs) != 0:
            raise ValueError(f'Fidelity parameters {", ".join(f_in_kwargs)} should not be part of kwargs\n'
                             f'Fidelity: {fidelity}\n Kwargs: {kwargs}')

        default_fidelities = fidelity_space.get_default_configuration()

        if fidelity is None:
            fidelity = default_fidelities
        if isinstance(fidelity, dict):
            default_fidelities_cfg = default_fidelities.get_dictionary()
            fidelity_copy = fidelity.copy()
            fidelity = {k: fidelity_copy.pop(k, v) for k, v in default_fidelities_cfg.items()}
            assert len(fidelity_copy) == 0, 'Provided fidelity dict contained unknown fidelity ' \
                                            f'values: {fidelity_copy.keys()}'
            fidelity = ConfigSpace.Configuration(fidelity_space, fidelity)
        elif isinstance(fidelity, ConfigSpace.Configuration):
            fidelity = fidelity
        else:
            raise TypeError(f'Fidelity has to be an instance of type None, dict, or '
                            f'ConfigSpace.Configuration but was {type(fidelity)}')
        # Ensure that the extracted fidelity values play well with the defined fidelity space
        fidelity_space.check_configuration(fidelity)
        return fidelity

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
