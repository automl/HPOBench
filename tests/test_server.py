import pytest
import os
import logging
import importlib


def test_debug_env_variable_1():
    os.environ['HPOLIB_DEBUG'] = 'false'
    from hpolib.container.client_abstract_benchmark import log_level
    assert log_level == logging.INFO

    os.environ['HPOLIB_DEBUG'] = 'true'
    import hpolib.container.client_abstract_benchmark as client
    importlib.reload(client)
    from hpolib.container.client_abstract_benchmark import log_level
    assert log_level == logging.DEBUG


if __name__ == '__main__':
    test_debug_env_variable_1()
