import logging
import time
from typing import Sequence, Callable, Optional, Dict, Union
import numpy as np
from hpobench.abstract_benchmark import AbstractBenchmark
import ConfigSpace

_log = logging.getLogger(__name__)


def sleep_objective_fn(obj, configuration: Dict,
                       fidelity: Union[Dict, None] = None,
                       rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
    """ A dummy objective function that does not really perform any computation but simply introduces a delay defined
    by the attribute '_sleep_duration' of the calling object. It returns a random value in [0, 1) for the key
    'function_value' and the value of 'obj_sleep_duration' for the cost. It also attaches a dictionary to the key
    'info' containing the rng, configuration and fidelity values the function was called with in the keys 'rng',
    'config' and 'fidelity' respectively. """

    _log.debug("Dummy objective function 'sleep_objective_fn' called.")
    for kwarg in kwargs.items():
        _log.debug("Ignoring additional keyword argument %s to objective function call." % str(kwarg))

    time.sleep(obj._sleep_duration)
    return {
        'function_value': np.random.random(),
        'cost': obj._sleep_duration,
        'info': {
            'rng': rng,
            'config': configuration,
            'fidelity': fidelity
        }
    }


class DummyBenchmark(AbstractBenchmark):
    def __init__(self,
                 configspace_params: Sequence[ConfigSpace.hyperparameters.Hyperparameter],
                 fidelity_params: Sequence[ConfigSpace.hyperparameters.Hyperparameter],
                 sleep_duration: Optional[float] = 1.0,
                 obj_fn: Optional[Callable] = sleep_objective_fn, rng: Union[int, np.random.RandomState, None] = None,
                 **kwargs):
        """
        Initializes a DummyBenchmark object with the specified characteristics. DummyBenchmark objects are not
        expected to perform any actual computations but instead serve as aids in debugging interfaces for HPOBench.
        They should be configured to expose characteristics expected of an actual benchmark. Note that DummyBenchmark
        objects are only useful after the object has been initialized and therefore are not suitable for tests designed
        to work on pre-initialization class methods only, such as those for calls to 'get_configuration_space()' on a
        Benchmark class itself. To this end, 'get_configuration_space()', 'get_fidelity_space()' and
        'get_meta_information()' have been re-defined as instance functions of the initialized object instead.

        Parameters
        ----------
        configspace_params: Sequence of ConfigSpace.hyperparameter.Hyperparameter objects
            This list of Hyperparameter objects is directly used to define the DummyBenchmark object's configuration
            space.
        fidelity_params: Sequence of ConfigSpace.hyperparameter.Hyperparameter objects
            This list of Hyperparameter objects is directly used to define the DummyBenchmark object's fidelity space.
        sleep_duration: float (optional)
            By default, the objective function of DummyBenchmark does not perform any computation and simply
            introduces a delay of 'sleep_duration' seconds. The default value is 1.0 (seconds).
        obj_fn: a callable object (optional)
            This enables the objective function of the dummy benchmark to be changed to a user specified function
            instead. Default: sleep_objective_fn.
        rng: int, np.random.RandomState, None
            The default random state for the benchmark. If type is int, a np.random.RandomState with seed `rng` is
            created. If type is None, create a new random state. Ignored by DummyBenchmark, directly passed to
            AbstractBenchmark.
        """

        _log.debug("Initializing DummyBenchmark object with %s configuration space parameters and %d fidelity space "
                   "parameters." % (len(configspace_params), len(fidelity_params)))
        self._sleep_duration = sleep_duration
        self._config_params = configspace_params
        self._fidelity_params = fidelity_params
        self._obj_fn = obj_fn
        self._meta = {'sleep_duration': sleep_duration, 'obj_fn': obj_fn.__name__}
        super(DummyBenchmark, self).__init__(rng=rng)
        _log.info("DummyBenchmark initializer received additional keyword arguments:\n%s" % str(kwargs))

    def get_configuration_space(self, seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        csp = ConfigSpace.ConfigurationSpace(name="Dummy Configuration Space", seed=seed)
        csp.add_hyperparameters(self._config_params)
        return csp

    def get_fidelity_space(self, seed: Union[int, None] = None) -> ConfigSpace.ConfigurationSpace:
        fsp = ConfigSpace.ConfigurationSpace(name="Dummy Fidelity Space", seed=seed)
        fsp.add_hyperparameters(self._fidelity_params)
        return fsp

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[ConfigSpace.Configuration, Dict],
                           fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        return self._obj_fn(self, configuration, fidelity, rng, **kwargs)

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[ConfigSpace.Configuration, Dict],
                                fidelity: Union[Dict, ConfigSpace.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        return self._obj_fn(self, configuration, fidelity, rng, **kwargs)

    def get_meta_information(self) -> Dict:
        return self._meta
