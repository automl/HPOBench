
import json

import gym
import numpy as np
import enum

from typing import Any, Union, List
from gym.spaces import Box, Discrete, Tuple, MultiDiscrete, MultiBinary, Dict

class Encoder(json.JSONEncoder):
    """ Json Encoder to save tuple and or numpy arrays | numpy floats / integer.
    Adapted from: https://github.com/automl/HPOBench/blob/master/hpobench/util/container_utils.py
    Serializing tuple/numpy array may not work. We need to annotate those types, to reconstruct them correctly.
    """

    @staticmethod
    def hint(item):
        # Annotate the different item types
        if isinstance(item, tuple):
            return {'__type__': 'tuple', '__items__': [Encoder.hint(e) for e in item]}
        if isinstance(item, np.ndarray):
            return {'__type__': 'np.ndarray', '__items__': item.tolist()}
        if isinstance(item, np.floating):
            return {'__type__': 'np.float', '__items__': float(item)}
        if isinstance(item, np.integer):
            return {'__type__': 'np.int', '__items__': item.tolist()}
        if isinstance(item, enum.Enum):
            return str(item)
        if isinstance(item, np.random.RandomState):
            return serialize_random_state(item)

        if isinstance(item, gym.Space):
            return Encoder.encode_space(item)
        if isinstance(item, np.dtype):
            return {'__type__': 'np.dtype', '__items__': str(item)}

        # If it is a container data structure, go also through the items.
        if isinstance(item, list):
            return [Encoder.hint(e) for e in item]
        if isinstance(item, dict):
            return {key: Encoder.hint(value) for key, value in item.items()}
        return item
    # pylint: disable=arguments-differ
    def encode(self, obj):
        return super(Encoder, self).encode(Encoder.hint(obj))

    @staticmethod
    def encode_space(space_obj : gym.Space):
        properties = [(
            '__type__',
            '.'.join([space_obj.__class__.__module__, space_obj.__class__.__name__]
                     )
        )]

        if not isinstance(space_obj, (gym.spaces.Dict, gym.spaces.Tuple)):
            properties.append(('np_random', serialize_random_state(space_obj.np_random)))

        if isinstance(space_obj, (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
            # by default assume all constrcutor arguments are stored under the same name
            #  for box we need to drop shape, since either shape or a array for low and height  is required
            __init__ = space_obj.__init__.__func__.__code__
            local_vars = __init__.co_varnames

            # drop self and non-args (self, arg1, arg2, ..., local_var1, local_var2, ...)
            arguments = local_vars[1:__init__.co_argcount]
            attributes_to_serialize = list(filter(lambda att: att not in ['shape'], arguments))

            for attribute in attributes_to_serialize:
                if hasattr(space_obj, attribute):
                    properties.append((attribute, Encoder.hint(getattr(space_obj, attribute))))
        elif isinstance(space_obj, gym.spaces.Tuple):
            properties.append(('spaces', [Encoder.encode_space(space) for space in space_obj.spaces]))
        elif isinstance(space_obj, gym.spaces.Dict):
            properties.append(('spaces', {name:Encoder.encode_space(space) for name, space in space_obj.spaces.items()}))
        else:
            raise NotImplemented(f"Serialisation for type {properties['__type__']} not implemented")

        return dict(properties)

class Decoder(json.JSONDecoder):
    """
    Adapted from: https://github.com/automl/HPOBench/blob/master/hpobench/util/container_utils.py
    """
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
                return deserialize_random_state(obj)
            if __type == 'np.dtype':
                return np.dtype(obj['__items__'])
            if __type.startswith('gym.spaces.'):
                return self.decode_space(obj)
        return obj


    def decode_space(self, space_dict) -> gym.Space:
        _type = space_dict['__type__']
        _class = getattr(gym.spaces, _type.split('.')[-1])
        print(_type, _class)

        args = {name: value for name, value in space_dict.items() if name not in ['__type__', 'np_random', 'shape']}

        if issubclass(_class, (gym.spaces.Dict, gym.spaces.Tuple, gym.spaces.tuple.Tuple)):
            pass

        space_object = _class(**args)

        if isinstance(space_object, gym.spaces.Tuple):
            space_object.spaces = tuple(space_object.spaces)

        if not isinstance(space_object, (gym.spaces.Dict, gym.spaces.Tuple, gym.spaces.tuple.Tuple)):
            space_object.np_random = space_dict['np_random']
        else:
            print("Warning(Restoring of nested random generator does not work reliable)")
        return space_object


def deserialize_random_state(random_state_dict) -> np.random.RandomState:
    (rnd0, rnd1, rnd2, rnd3, rnd4) = random_state_dict['__items__']
    rnd1 = np.array(rnd1, dtype=np.uint32)
    random_state = np.random.RandomState()
    random_state.set_state((rnd0, rnd1, rnd2, rnd3, rnd4))
    return random_state

def serialize_random_state(random_state: np.random.RandomState):
    (rnd0, rnd1, rnd2, rnd3, rnd4) = random_state.get_state()
    rnd1 = rnd1.tolist()
    return {'__type__': 'random_state', '__items__': [rnd0, rnd1, rnd2, rnd3, rnd4]}


if __name__ == '__main__':
    tuple_space = Tuple((Box(low=-1, high=1, shape=(2,)), Box(low=-1, high=1, shape=(2,))))
    dict_space = Dict({'a': Box(low=-1, high=1, shape=(2,)), 'b': MultiBinary([2, 2])})

    print(serialize_random_state(tuple_space.spaces[0].np_random))

    spaces = [tuple_space, dict_space]

    for space in spaces:
        serialized = json.dumps(space, cls=Encoder)
        restored_space = json.loads(serialized, cls=Decoder)
        assert space == restored_space
        s1 = space.sample()
        s2 = restored_space.sample()
        assert s1 == s2