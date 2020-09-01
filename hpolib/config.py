import ast
import logging
import os
from pathlib import Path

import yaml

from hpolib import __version__


class HPOlibConfig:

    def __init__(self):
        """ Class holding the configuration for the HPOlib. When initialized, it reads (or creates)
        the config file and data directory accordingly.

        Parameters:
        -----------

        config_file : str
           Path to config file
        """
        self.logger = logging.getLogger('HPOlibConfig')

        # According to https://github.com/openml/openml-python/issues/884, try to set default directories.
        config_base_dir = Path(os.environ.get('XDG_CONFIG_HOME', '~/.config/hpolib2')).expanduser()

        # The path to the hpolibrc file
        self.config_file = config_base_dir / '.hpolibrc'
        self.config_file = self.config_file.expanduser().absolute()

        # The configuration file should have the same version as the hpolib. Raise a warning if there is a version
        # mismatch. We want to make sure that the configuration file is up to date with the hpolib, because it is
        # possible that something in the configuration has changed but the old hpolibrc file is still there and prohibit
        # the changes. Ignore the dev tag if available.
        self.config_version = __version__.rstrip('dev')

        # Set the default logging level.
        # Possible levels are 0 = NOTSET (warning), 10 = DEBUG, 20 = INFO, 30 = WARNING, 40 = ERROR, 50 = CRITICAL
        # See also https://docs.python.org/3/library/logging.html#logging-levels
        self.verbosity = 0

        # The cache dir contains the lock files etc
        self.cache_dir = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache/hpolib2')).expanduser()

        # The user can specify if the local or a more global data directory should be used. This is helpful when working
        # on a cluster
        self.data_dir = config_base_dir
        self.global_data_dir = Path(os.environ.get('XDG_DATA_HOME', '~/.local/share/hpolib2')).expanduser()
        self.use_global_data = True

        # Options for the singularity container
        # Find all hosted container on: https://cloud.sylabs.io/library/phmueller/automl
        self.socket_dir = Path('/tmp')
        self.container_dir = self.cache_dir / f'hpolib2-{os.getuid()}'
        self.container_source = 'library://phmueller/automl'
        self.pyro_connect_max_wait = 400

        # Read in the hpolibrc file and set the default values if not specified
        self._setup()

    def _setup(self):
        """ Sets up config. Reads the config file and parses it.

        Parameters:
        -----------

        config_file: Path, str
            Path to config file
        """

        # Create an empty config file if there was none so far
        if not self.config_file.exists():
            self.__create_config_file()

        # Parse config
        self.__parse_config()

        # Check whether data_dir exists, if not create it
        self.__check_dir(self.data_dir)
        self.__check_dir(self.global_data_dir)
        self.__check_dir(self.socket_dir)
        self.__check_dir(self.container_dir)
        self.__check_dir(self.cache_dir)

    def __create_config_file(self):
        """ Create the configuration file. """
        self.logger.debug(f'Create a new config file here: {self.config_file}')
        self.__check_dir(self.config_file.parent)

        defaults = {'version': self.config_version,
                    'verbosity': self.verbosity,
                    'cache_dir': str(self.cache_dir),
                    'data_dir': str(self.data_dir),
                    'global_data_dir': str(self.global_data_dir),
                    'use_global_data': True,
                    'socket_dir': str(self.socket_dir),
                    'container_dir': str(self.container_dir),
                    'container_source': self.container_source,
                    'pyro_connect_max_wait': self.pyro_connect_max_wait
                    }

        with self.config_file.open('w', encoding='utf-8') as fh:
            yaml.dump(defaults, fh)

    def __parse_config(self):
        """ Parse the config file """
        with self.config_file.open('r') as fh:
            read_config = yaml.load(fh, Loader=yaml.FullLoader)

        # The old hpolibrc was parsed with the configparser. But this required to use fake sections, etc. We moved to
        # pyyaml. Yaml returns a string if the rc file is not in yaml format.
        if isinstance(read_config, str):
            logging.warning('The hpolibrc can not be parsed. This is likely due to a change in the hpolibrc format.'
                            f' Please remove the old hpolibrc. The hpolibrc file is in {self.config_file}')

        self.config_version = read_config.get('version')
        self._check_version()

        self.verbosity = read_config.get('verbosity', self.verbosity)
        logging.basicConfig(level=self.verbosity)

        self.cache_dir = Path(read_config.get('cache_dir', self.cache_dir))
        self.data_dir = Path(read_config.get('data_dir', self.data_dir))
        self.global_data_dir = Path(read_config.get('global_data_dir', self.global_data_dir))
        self.use_global_data = read_config.get('use_global_data', self.use_global_data)

        if isinstance(self.use_global_data, str):
            self.use_global_data = ast.literal_eval(self.use_global_data)

        if self.global_data_dir.is_dir() and self.use_global_data:
            self.data_dir = self.global_data_dir

        self.socket_dir = Path(read_config.get('socket_dir', self.socket_dir))
        self.container_dir = Path(read_config.get('container_dir', self.container_dir))
        self.container_source = read_config.get('container_source', self.container_source)
        self.pyro_connect_max_wait = int(read_config.get('pyro_connect_max_wait',
                                                         self.pyro_connect_max_wait))

    def _check_version(self):
        """ Check if the version of the configuration file matches the hpolib version.
            Ignore the `dev` tag at the end.
        """
        if self.config_version is None or not __version__.startswith(self.config_version):
            self.config_version = self.config_version if not None else 'None'
            logging.warning(f'The hpolibrc file was created with another version of the hpolib. '
                            f'Current version of the hpolibrc file: {self.config_version}.\n'
                            f'Current version of the hpolib: {__version__}')

    def __check_dir(self, path: Path):
        """ Check whether dir exists and if not create it"""
        try:
            Path(path).mkdir(exist_ok=True, parents=True)
        except (IOError, OSError) as e:
            self.logger.debug(f'Could not create directory here: {self.data_dir}')
            raise e


config_file = HPOlibConfig()
__all__ = ['config_file', 'HPOlibConfig']
