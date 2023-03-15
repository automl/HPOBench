import os

CONST_RUN_ALL_TESTS_ENV_VAR = 'HPOBENCH_RUN_EXPENSIVE_TESTS'
DEFAULT_SKIP_MSG = 'Skip this test due to time limitations'


def check_run_all_tests():
    """ Helper function: Check if all tests should run. """
    return os.environ.get(CONST_RUN_ALL_TESTS_ENV_VAR, 'false').lower() == 'true'


def enable_all_tests():
    """
    Some tests are quite expensive. We control if all runs should be executed by this
    environment variable.
    """
    os.environ[CONST_RUN_ALL_TESTS_ENV_VAR] = 'true'


def disable_all_tests():
    """
    This function disables the evaluation of all test functions.
    """
    os.environ[CONST_RUN_ALL_TESTS_ENV_VAR] = 'false'
