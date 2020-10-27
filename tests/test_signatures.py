import inspect
from pathlib import Path
import importlib

params_init = ['rng']
params_config_space = ['seed']
params_obj_func = ['self', 'configuration', 'fidelity', 'shuffle', 'rng', 'kwargs']

found_exception = False


def valid_signature(func_name, params):
    if func_name == 'objective_function':
        signature = inspect.signature(c[1].objective_function).parameters.keys()
    elif func_name == 'objective_function_test':
        signature = inspect.signature(c[1].objective_function_test).parameters.keys()
    elif func_name == 'get_configuration_space':
        signature = inspect.signature(c[1].get_configuration_space).parameters.keys()
    elif func_name == 'init':
        signature = inspect.signature(c[1].__init__).parameters.keys()
    else:
        raise ValueError('Unknown func name')

    try:
        assert all([param in keys for param in params_obj_func]), \
            f'Benchmark {c} does not implement the {func_name} signature correctly. ' \
            f'Found parameters: {signature} but expected: {params}'
        return False
    except AssertionError:
        return True


module = importlib.import_module(f'hpolib.container.benchmarks')
benchmarks = list(f for f in Path(module.__file__).parent.rglob('*.py') if f.name != '__init__.py')

for container_str in ['', '.container']:
    for benchmark in benchmarks:
        try:
            module = importlib.import_module(f'hpolib{container_str}.benchmarks.{benchmark.parent.name}.{benchmark.name.replace(".py", "")}')
        except ImportError as e:
            # Todo: This happens when the module was not importable. For example if a package is missing.
            #       Perhaps there is a better way to test this?
            print(f'WARNING: Can\'t import {benchmark}. Error message was: {e}')
            continue

        classes = inspect.getmembers(module, inspect.isclass)
        classes = [c for c in classes if 'hpolib{container_str}.benchmark' in str(c)]

        for c in classes:
            valid_obj_func = valid_signature(func_name='objective_function', params=params_obj_func)
            valid_obj_func_test = valid_signature(func_name='objective_function_test', params=params_obj_func)
            valid_cs = valid_signature(func_name='get_configuration_space', params=params_config_space)
            valid_init = valid_signature(func_name='init', params=params_init)
            found_exception = found_exception or not (valid_obj_func and valid_obj_func_test and valid_cs and valid_init)

benchmarks = list(f for f in Path(module.__file__).parent.rglob('*.py') if f.name != '__init__.py')

if found_exception:
    raise AssertionError('Not all signatures where correct.')
