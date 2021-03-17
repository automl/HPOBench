"""
Changelog:
==========
0.0.7:
* Update the version check:  Instead of requesting the same version, we check if the configuration file version and the
  hpobench version are in the same partition. Each hpobench version that has a compatible configuration file definition
  is in the same distribution.

0.0.6:
* Rewrote the configuration (=hpobenchrc) file: We are now using a yaml structure.
* Check if the hpobenchrc file is created with a the same hpobench version.

0.0.1:
* Configuration file based on configparser

"""


import ast
import logging
import os
from pathlib import Path

import yaml
from yaml.parser import ParserError

from hpobench import __version__


class HPOBenchConfig:

    def __init__(self):
        """ Class holding the configuration for the HPOBench. When initialized, it reads (or creates)
        the config file and data directory accordingly.

        Parameters:
        -----------

        config_file : str
           Path to config file
        """
        self.logger = logging.getLogger('HPOBenchConfig')

        # According to https://github.com/openml/openml-python/issues/884, try to set default directories.
        self.config_base_dir = Path(os.environ.get('XDG_CONFIG_HOME', '~/.config/hpobench')).expanduser()

        # The path to the hpobenchrc file
        self.config_file = self.config_base_dir / '.hpobenchrc'
        self.config_file = self.config_file.expanduser().absolute()

        # The configuration file should have the same version as the hpobench. Raise a warning if there is a version
        # mismatch. We want to make sure that the configuration file is up to date with the hpobench, because it is
        # possible that something in the configuration has changed but the old hpobenchrc file is still there and
        # prohibit the changes. Ignore the dev tag if available.
        self.config_version = __version__.rstrip('dev')

        # Set the default logging level.
        # Possible levels are 0 = NOTSET (warning), 10 = DEBUG, 20 = INFO, 30 = WARNING, 40 = ERROR, 50 = CRITICAL
        # See also https://docs.python.org/3/library/logging.html#logging-levels
        self.verbosity = 0

        # The cache dir contains the lock files etc
        self.cache_dir = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache/hpobench')).expanduser()

        # The user can specify if the local or a more global data directory should be used. This is helpful when working
        # on a cluster
        self.data_dir = self.config_base_dir
        self.global_data_dir = Path(os.environ.get('XDG_DATA_HOME', '~/.local/share/hpobench')).expanduser()
        self.use_global_data = True

        # Options for the singularity container
        # Find all hosted container on: https://cloud.sylabs.io/library/phmueller/automl
        self.socket_dir = Path('/tmp')
        self.container_dir = self.cache_dir / f'hpobench-{os.getuid()}'
        self.container_source = 'library://phmueller/automl'
        self.pyro_connect_max_wait = 400

        # Read in the hpobenchrc file and set the default values if not specified
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
        failure_msg = 'The hpobenchrc can not be parsed. This is likely due to a change in the hpobenchrc format.'\
                      f' Please remove the old hpobenchrc and restart the procedure. ' \
                      f'The hpobenchrc file is in {self.config_file}'
        try:
            with self.config_file.open('r') as fh:
                read_config = yaml.load(fh, Loader=yaml.FullLoader)
        except ParserError:
            raise ParserError(failure_msg)

        # The old hpolibrc was parsed with the configparser. But this required to use fake sections, etc. We moved to
        # pyyaml. Yaml returns a string if the rc file is not in yaml format.
        if isinstance(read_config, str):
            raise ParserError(failure_msg)

        self.config_version = read_config.get('version')
        self._check_version(self.config_version, __version__)

        self.verbosity = read_config.get('verbosity', self.verbosity)
        # logging.basicConfig(level=self.verbosity)  # TODO: This statement causes some trouble.
        #                                                    (Container-Logger is always set to debug)

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

    @staticmethod
    def _check_version(config_version, hpobench_version):
        """ Check if the version of the configuration file matches the hpobench version.
            Ignore the `dev` tag at the end.

            It may happen that the configuration file changes with a new version of the hpobench. In this case,
            we want to raise a warning to show the user that there could be potential error.

            But since the configuration file does not change every new hpobench version, we group versions together that
            have similar configuration files. If the current version is equal to a version from the same partition, then
            no problem can occur.
        """
        version_partitions = [['0.0.0', '0.0.5'],
                              ['0.0.6', '0.0.7'],
                              ['0.0.8', '999.999.999']]

        mismatch = False

        if config_version is None:
            mismatch = True
        else:
            def __int_representation(version_number: str):
                # Convert a string with the format 'xxx.xxx.xxx' to a comparable number.
                # Multiply each part from left to right with 10^6, 10^3, 10^0.
                # we allow here 1000 versions until a next release (*) has to happen. This should be more than enough.
                # *) new release means increase of a more left number: e.g. 0.0.2 --> 0.1.0
                version_number = version_number.replace('_', '').replace('dev', '')

                value = [int(v) for v in version_number.split('.')]
                return value[0] * 10**6 + value[1] * 10**3 + value[2]

            config_version_ = __int_representation(config_version)
            hpobench_version_ = __int_representation(hpobench_version)

            for lower, upper in version_partitions:
                # Test if the configuration version and the hpobench version are in the same partition.
                # We test here by converting the version number to an ordinal number scale. Then, we can compare the
                # version numbers with range comparisons.

                lower = __int_representation(lower)
                upper = __int_representation(upper)

                if lower <= config_version_ <= upper and lower <= hpobench_version_ <= upper:
                    break
            else:
                mismatch = True

        if mismatch:
            config_version = config_version if not None else 'None'
            logging.warning(f'The hpobenchrc file was created with another version of the hpobench. '
                            f'Current version of the hpobenchrc file: {config_version}.\n'
                            f'Current version of the hpobench: {hpobench_version}')

        return not mismatch

    def __check_dir(self, path: Path):
        """ Check whether dir exists and if not create it"""
        try:
            Path(path).mkdir(exist_ok=True, parents=True)
        except (IOError, OSError) as e:
            self.logger.debug(f'Could not create directory here: {self.data_dir}')
            raise e


config_file = HPOBenchConfig()
__all__ = ['config_file', 'HPOBenchConfig']
