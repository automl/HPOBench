import ast
import configparser
import logging
import os
from io import StringIO
from pathlib import Path
from typing import Union, Any


class HPOlibConfig:

    def __init__(self):
        """ Holds configuration for HPOlib. When initialized reads (or creates)
         the config file and data directory accordingly.

         Parameters:
         -----------

        config_file: str
            Path to config file
        """
        self.logger = logging.getLogger('HPOlibConfig')

        # According to https://github.com/openml/openml-python/issues/884,
        # try to set default directories.
        self.config_base_dir = \
            Path(os.environ.get('XDG_CONFIG_HOME',
                                '~/.config/hpolib3')).expanduser()
        self.config_file = self.config_base_dir / '.hpolibrc'
        self.cache_dir = Path(os.environ.get('XDG_CACHE_HOME',
                                             '~/.cache/hpolib3')).expanduser()

        # self.global_data_dir = Path('/var/lib/hpolib3/')
        self.global_data_dir = \
            Path(os.environ.get('XDG_DATA_HOME',
                                '~/.local/share/hpolib3')).expanduser()

        self.config = None
        self.data_dir = None
        self.socket_dir = None
        self.container_source = None
        self.use_global_data = None

        self.defaults = \
            {'verbosity': 0,
             'data_dir': self.config_base_dir,
             'socket_dir': self.cache_dir,
             'container_dir': self.cache_dir / f'hpolib3-{os.getuid()}',
             'container_source': 'shub://automl/HPOlib3',
             'use_global_data': True,
             'pyro_connect_max_wait': 60}

        self._setup(self.config_file)

    def _setup(self, config_file: Union[Path, str]):
        """ Sets up config. Reads the config file and parses it.

        Parameters:
        -----------

        config_file: Path, str
            Path to config file
        """

        # Change current config file to new config file
        config_file = Path(config_file).expanduser().absolute()

        if config_file != self.config_file:
            self.logger.debug(f'Change config file from {self.config_file} '
                              f'to {config_file}')
            self.config_file = config_file

        # Create an empty config file if there was none so far
        if not self.config_file.exists():
            self.__create_config_file()

        # Parse config and store input in self.config
        self.__parse_config()

        # Check whether data_dir exists, if not create it
        self.__check_dir(self.data_dir)
        self.__check_dir(self.socket_dir)
        self.__check_dir(self.container_dir)
        self.__check_dir(self.cache_dir)

    def __create_config_file(self):
        """ Create the configuration file. """
        try:
            self.logger.debug(f'Create a new config file here: '
                              f'{self.config_file}')
            self.__check_dir(self.config_file.parent)
            fh = self.config_file.open('w', encoding='utf-8')
            for k in self.defaults:
                fh.write(f'{k}={self.defaults[k]}\n')
            fh.close()
        except (IOError, OSError):
            raise

    def __parse_config(self):
        """ Parse the config file """
        config = configparser.RawConfigParser()

        # Cheat the ConfigParser module by adding a fake section header
        config_file_ = StringIO()
        config_file_.write('[FAKE_SECTION]\n')
        with self.config_file.open('r', encoding='utf-8') as fh:
            for line in fh:
                config_file_.write(line)
        config_file_.seek(0)
        config.read_file(config_file_)
        self.config = config

        # Store configuration
        self.data_dir = Path(self.__get_config_option('data_dir'))
        self.socket_dir = Path(self.__get_config_option('socket_dir'))
        self.container_dir = Path(self.__get_config_option('container_dir'))
        self.container_source = self.__get_config_option('container_source')
        self.use_global_data = self.__get_config_option('use_global_data')
        if type(self.use_global_data) is str:
            self.use_global_data = ast.literal_eval(self.use_global_data)
        self.pyro_connect_max_wait = \
            int(self.__get_config_option('pyro_connect_max_wait'))

        # Use global data dir if exist
        if self.global_data_dir.is_dir() and self.use_global_data:
            self.data_dir = self.global_data_dir

    def __get_config_option(self, o: str) -> Any:
        """ Try to get config option from configuration file. If the option is
            not configured, use default """
        try:
            return self.config.get('FAKE_SECTION', o)
        except configparser.NoOptionError:
            return self.defaults[o]

    def __check_dir(self, path: Path):
        """ Check whether dir exists and if not create it"""
        try:
            Path(path).mkdir(exist_ok=True)
        except (IOError, OSError) as e:
            self.logger.debug(f'Could not create directory here: '
                              f'{self.data_dir}')
            raise e


config_file = HPOlibConfig()
__all__ = ['config_file', 'HPOlibConfig']
