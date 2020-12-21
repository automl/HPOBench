import os
import importlib
import json
import numpy as np
import enum

from typing import Any, Union

from hpobench.util.rng_helper import serialize_random_state, deserialize_random_state


class BenchmarkEncoder(json.JSONEncoder):
    """ Json Encoder to save tuple and or numpy arrays | numpy floats / integer.
    from: https://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json

    Serializing tuple/numpy array may not work. We need to annotate those types, to reconstruct them correctly.
    """
    # pylint: disable=arguments-differ
    def encode(self, obj):
        def hint(item):
            # Annotate the different item types
            if isinstance(item, tuple):
                return {'__type__': 'tuple', '__items__': [hint(e) for e in item]}
            if isinstance(item, np.ndarray):
                return {'__type__': 'np.ndarray', '__items__': item.tolist()}
            if isinstance(item, np.floating):
                return {'__type__': 'np.float', '__items__': float(item)}
            if isinstance(item, np.integer):
                return {'__type__': 'np.int', '__items__': item.tolist()}
            if isinstance(item, enum.Enum):
                return str(item)
            if isinstance(item, np.random.RandomState):
                rs = serialize_random_state(item)
                return {'__type__': 'random_state', '__items__': rs}

            # If it is a container data structure, go also through the items.
            if isinstance(item, list):
                return [hint(e) for e in item]
            if isinstance(item, dict):
                return {key: hint(value) for key, value in item.items()}
            return item

        return super(BenchmarkEncoder, self).encode(hint(obj))


class BenchmarkDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Any) -> Union[Union[tuple, np.ndarray, float, float, int], Any]:
        if '__type__' in obj:
            __type = obj['__type__']

            if __type == 'tuple':
                return tuple(obj['__items__'])
            if __type == 'np.ndarray':
                return np.array(obj['__items__'])
            if __type == 'np.float':
                return np.float(obj['__items__'])
            if __type == 'np.int':
                return np.int(obj['__items__'])
            if __type == 'random_state':
                return deserialize_random_state(obj['__items__'])
        return obj


def __reload_module():
    """
    The env variable which enables the debug level is read in during the import of the client module.
    Reloading the module, re-reads the env variable and therefore changes the level.
    """
    import hpobench.container.client_abstract_benchmark as client
    importlib.reload(client)


def enable_container_debug():
    """ Sets the environment variable "HPOBENCH_DEBUG" to true. The container checks this variable and if set to true,
        enables debugging on the container side. """
    os.environ['HPOBENCH_DEBUG'] = 'true'
    __reload_module()


def disable_container_debug():
    os.environ['HPOBENCH_DEBUG'] = 'false'
    __reload_module()
