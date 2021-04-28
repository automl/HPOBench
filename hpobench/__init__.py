import logging

_default_log_format = '[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'
logging.basicConfig(format=_default_log_format, level=logging.WARNING)
root_logger = logging.getLogger()

from hpobench.__version__ import __version__  # noqa: F401, E402
from hpobench.config import config_file  # noqa: F401, E402

__contact__ = "automl.org"
