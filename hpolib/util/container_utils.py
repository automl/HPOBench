import os


def enable_container_debug():
    """ Sets the environment variable "HPOLIB_DEBUG" to true. The container checks this variable and if set to true,
        enables debugging on the container side. """
    os.environ['HPOLIB_DEBUG'] = 'true'


def disable_container_debug():
    os.environ['HPOLIB_DEBUG'] = 'false'
