#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the client-side for using benchmarks with containers

AbstractBenchmarkClient defines the client side for the communication between
the containers and the client.
It is used to download (if not already) and start containers in the background.

To reduce download traffic, firstly, it checks if the container is already
downloaded. The container source as well as the path, where it should be stored,
are defined in the ~/.hpobenchrc - file.

The name of the container (``container_name``) is defined either in its belonging
container-benchmark definition. (hpobench/container/<type>/<name> or via ``container_name``.
"""
import os
import abc
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional
from typing import Union, Dict
from uuid import uuid1

import ConfigSpace as CS
import Pyro4
import Pyro4.util
import Pyro4.errors
import numpy as np
from ConfigSpace.read_and_write import json as csjson
from oslo_concurrency import lockutils

import hpobench.config
from hpobench import __version__
from hpobench.util.container_utils import BenchmarkEncoder, BenchmarkDecoder

# Read in the verbosity level from the environment variable HPOBENCH_DEBUG
log_level_str = os.environ.get('HPOBENCH_DEBUG', 'false')
LOG_LEVEL = logging.DEBUG if log_level_str == 'true' else logging.INFO

root = logging.getLogger()
root.setLevel(level=LOG_LEVEL)

logger = logging.getLogger("BenchmarkClient")
logger.setLevel(level=LOG_LEVEL)

# This option improves the quality of stacktraces if a container crashes
sys.excepthook = Pyro4.util.excepthook

# Number of tries to connect to server
MAX_TRIES = 5


class AbstractBenchmarkClient(metaclass=abc.ABCMeta):
    """ Base Class for the containerized benchmarks.

    Attributes
    ----------
    socket_id : str
    """
    def __init__(self, benchmark_name: str, container_name: str, container_source: Optional[str] = None,
                 container_tag: str = 'latest', env_str: Optional[str] = '', bind_str: Optional[str] = '',
                 gpu: Optional[bool] = False, rng: Union[np.random.RandomState, int, None] = None,
                 socket_id=None, **kwargs):

        self.config = hpobench.config.config_file

        # connect to already running server if a socket_id is given. In this case, skip the init of
        # the benchmark
        self.proxy_only = socket_id is not None

        if not self.proxy_only:
            self.socket_id = self._id_generator()
            self.load_benchmark(benchmark_name, container_name, container_source, container_tag, env_str, bind_str, gpu, rng,
                                **kwargs)
            self.start_server(benchmark_name, container_name, container_source, container_tag, env_str, bind_str, gpu, rng,
                              **kwargs)
            self.connect_to_server()
            self.init_benchmark(rng, **kwargs)
        else:
            self.socket_id = socket_id
            self.connect_to_server()

    def _parse_kwargs(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """ Helper function to parse the named keyword arguments to json str. """
        if rng is not None:
            kwargs['rng'] = rng
        if 'latest' in kwargs:
            del kwargs['latest']
        kwargs_str = json.dumps(kwargs, indent=None, cls=BenchmarkEncoder)
        return kwargs_str

    def _parse_configuration(self, configuration: Union[CS.Configuration, Dict]) -> str:
        if isinstance(configuration, CS.Configuration):
            configuration = configuration.get_dictionary()
        elif isinstance(configuration, dict):
            configuration = configuration
        else:
            raise ValueError(f'Type of config not understood: {type(configuration)}')
        c_str = json.dumps(configuration, indent=None, cls=BenchmarkEncoder)
        return c_str

    def _parse_fidelities(self, fidelity: Union[CS.Configuration, Dict, None] = None):
        if fidelity is None:
            fidelity = {}
        elif isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()
        elif isinstance(fidelity, dict):
            fidelity = fidelity
        else:
            raise ValueError(f'Type of fidelity not understood: {type(fidelity)}')
        f_str = json.dumps(fidelity, indent=None, cls=BenchmarkEncoder)
        return f_str

    def load_benchmark(self, benchmark_name: str, container_name: str, container_source: Optional[str] = None,
               container_tag: str = 'latest', env_str: Optional[str] = '', bind_str: Optional[str] = '',
               gpu: Optional[bool] = False, rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        # We can point to a different container source. See below.
        self.container_source = container_source or self.config.container_source
        self.container_dir = Path(self.config.container_dir)

        if (self.container_source.startswith('oras://gitlab.tf.uni-freiburg.de:5050/muelleph/hpobench-registry')
                and container_tag == 'latest'):
            assert 'latest' in kwargs, 'If the container is hosted on the gitlab registry, make sure that in the ' \
                                       'container init, the field \'latest\' is set.'

            container_tag = kwargs['latest']
            logger.debug(f'Replace the tag \'latest\' with \'{container_tag}\'.')

        self.container_name_with_tag = f'{container_name}_{container_tag}'
        logger.info(f'~~~ HPOBENCH VERSION: {__version__} ~~~~ CONTAINER VERSION: {container_tag} ~~~')
        logger.debug(f'Use benchmark {benchmark_name} from container {self.container_source}/{container_name}. \n'
                     f'And container directory {self.config.container_dir}')

        # Pull the container from the singularity hub if the container is hosted online. If the container is stored
        # locally (e.g.for development) do not pull it.
        if self.container_source is not None \
                and any((s in self.container_source for s in ['shub', 'library', 'docker', 'oras', 'http'])):

            # Racing conditions: If a process is already loading the benchmark container, let all other processes wait.
            # Following https://github.com/dhellmann/oslo.concurrency/blob/master/openstack/common/lockutils.py
            # (line 56), we don't need to handle any exception which can occur in the download_container-method.
            # The lock is released if the process crashes.
            # Also, oslo.concurrency does not delete the unused lockfiles
            # after usage. (An existing lock file does not mean that it is still locked!). They argue that in their
            # "testing, when a lock file was deleted while another process was waiting for it, it created a sort of
            # split-brain situation between any process that had been waiting for the deleted file, and any process
            # that attempted to lock the file after it had been deleted."
            # See: https://docs.openstack.org/oslo.concurrency/latest/admin/index.html
            @lockutils.synchronized('not_thread_process_safe', external=True,
                                    lock_path=f'{self.config.cache_dir}/lock_{container_name}', delay=5)
            def download_container(container_dir, container_name, container_source, container_tag):
                if not (container_dir / self.container_name_with_tag).exists():
                    logger.debug('Going to pull the container from an online source.')

                    container_dir.mkdir(parents=True, exist_ok=True)

                    cmd = f'singularity pull --dir {self.config.container_dir} ' \
                          f'--name {self.container_name_with_tag} '

                    # Currently, we can't see the available container tags on gitlab. Therefore, we create for each
                    # "tag" a new entry in the registry. This might change in the future. But as long as we don't have
                    # a fix for this, we need to map the container tag differently.
                    if container_source.startswith('oras://gitlab.tf.uni-freiburg.de:5050/muelleph/hpobench-registry'):
                        cmd += f'{container_source}/{container_name.lower()}/{container_tag}:latest'
                    else:
                        cmd += f'{container_source}/{container_name.lower()}:{container_tag}'

                    logger.info(f'Start downloading the container {self.container_name_with_tag} from {container_source}. '
                                'This may take several minutes.')
                    logger.debug(cmd)
                    subprocess.run(cmd, shell=True, check=True)
                    time.sleep(1)
                else:
                    logger.debug('Skipping downloading the container. It is already downloaded.')

            download_container(self.container_dir, container_name, self.container_source, container_tag)
        else:
            logger.debug(f'Looking on the local filesystem for the container file, since container source was '
                         f'either \'None\' or not a known address. Image Source: {self.container_source}')

            # Make sure that the container can be found locally.
            self.container_dir = Path(self.container_source)

            if not self.container_dir.exists():
                raise FileNotFoundError(f'Could not find the container on the local filesystem. The path '
                                        f'{self.container_source} does not exist.'
                                        'Please either specify the full path to the container '
                                        'or the directory in which the container is, as well as '
                                        'the benchmark name and the container tag (default: latest).')

            # if the container source is the path to the container itself, we are going to use this container directly.
            if self.container_dir.is_file():
                self.container_name_with_tag = self.container_dir.name
                self.container_dir = self.container_dir.parent

            # If the user specifies a container directory, search for the container name with (!) tag in it.
            elif self.container_dir.is_dir():
                assert (self.container_dir / self.container_name_with_tag).exists(), \
                    f'Local container not found in {self.container_dir / self.container_name_with_tag}'

            else:
                raise FileNotFoundError('The container source is neither a file nor a directory.'
                                        f'self.container_source: {self.container_dir}')

            logger.debug('Image found on the local file system.')

    def start_server(self, benchmark_name: str, container_name: str, container_source: Optional[str] = None,
               container_tag: str = 'latest', env_str: Optional[str] = '', bind_str: Optional[str] = '',
               gpu: Optional[bool] = False, rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        env_vars = {'HPOBENCH_DEBUG': log_level_str}
        if env_str.strip() != '':
            # Following the documentation of singularity, actually all env variables should have a
            # 'SINGULARITYENV_'-prefix. However, it works also without it. We want as environmental variables input
            # a string of form VAR1=VAL1,VAR2=VAL2,...
            env_str = env_str.replace(' ', '').rstrip(',')
            for k_v in env_str.split(','):
                k, v = k_v.split('=')
                env_vars[k] = v

        bind_options = f'--bind ' \
                       f'{self.config.cache_dir}:{self.config._cache_dir_container},' \
                       f'{self.config.data_dir}:{self.config._data_dir_container},' \
                       f'{self.config.socket_dir}:{self.config._socket_dir_container}'
        if bind_str.strip() != '':
            bind_options += ',' + bind_str.strip()

        gpu_opt = '--nv ' if gpu else ''  # Option for enabling GPU support

        assert self.container_dir is not None and self.container_name_with_tag is not None

        cmd = f'singularity instance start {bind_options} {gpu_opt}' \
              f'{self.container_dir / self.container_name_with_tag} {self.socket_id}'
        logger.debug(cmd)
        logger.debug(cmd)

        for num_tries in range(MAX_TRIES):
            p = subprocess.Popen(cmd,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            output, err = p.communicate()
            logger.debug(err)

            # Give a little bit of time to make sure the instance does not disappears after testing if it is there.
            time.sleep(1)

            # Check if the instance is started. Starting a instance crashes sometimes without a warning.
            # Therefore, try this step multiple times unless it is really started
            out = subprocess.getoutput('singularity instance list')
            out = out.split()

            if self.socket_id in out:
                break

            logger.debug(f'Could not start instance: Try {num_tries + 1}|{MAX_TRIES}')
            if num_tries + 1 == MAX_TRIES:
                raise SystemError(f'Could not start a instance of the benchmark. Retried {MAX_TRIES:d} times'
                                  f'\nStdout: {output} \nStderr: {err}')

            sleep_for = np.random.randint(1, 60)
            logger.critical(f'[{num_tries + 1}/{MAX_TRIES}] Could not start instance, sleeping for {sleep_for} seconds')
            time.sleep(sleep_for)

        # Give each instance a little bit time to start
        time.sleep(1)

        cmd = f'singularity run {gpu_opt}instance://{self.socket_id} {benchmark_name} {self.socket_id}'
        logger.debug(cmd)
        subprocess.Popen(cmd.split(), shell=False,
                         env={**os.environ, **{'SINGULARITYENV_HPOBENCH_DEBUG': log_level_str}})
        logger.debug('Instance successfully started')

    def connect_to_server(self):
        Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        self.uri = f'PYRO:{self.socket_id}.unixsock@./u:' \
                   f'{self.config.socket_dir}/{self.socket_id}_unix.sock'
        self.benchmark = Pyro4.Proxy(self.uri)
        logger.debug('Connected Proxy to benchmark')

    def init_benchmark(self, rng, **kwargs):
        # Handle rng and other optional benchmark arguments
        kwargs_str = self._parse_kwargs(rng, **kwargs)

        # Try to connect to server calling benchmark constructor via RPC. There exists a time limit
        logger.debug('Check connection to container and init benchmark')
        wait = 0
        while True:
            try:
                self.benchmark.init_benchmark(kwargs_str)
            except Pyro4.core.errors.CommunicationError:
                logger.debug('Still waiting')
                time.sleep(5)
                wait += 5
                if wait < self.config.pyro_connect_max_wait:
                    continue
                logger.debug('Waiting time exceeded. To increase, adjust config option pyro_connect_max_wait.')
                raise TimeoutError()

            break
        logger.debug('Connected to container')

    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Run a given configuration for a given fidelity on the containerized benchmark.

        Convert the given parameters to strings and send them via Pyro to the container.
        Read the result information and parse them.

        Parameters
        ----------
        configuration :CS.Configuration, Dict
        fidelity : CS.Configuration, Dict, None
        rng : np.random.RandomState, int, None
        kwargs : Dict

        Returns
        -------
        Dict
        """
        c_str = self._parse_configuration(configuration)
        f_str = self._parse_fidelities(fidelity)
        kwargs_str = self._parse_kwargs(rng, **kwargs)

        json_str = self.benchmark.objective_function(c_str, f_str, kwargs_str)
        return json.loads(json_str, cls=BenchmarkDecoder)

    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Run a given configuration for a given fidelity on the test function of the containerized  benchmark.

        Convert the given parameters to strings and send them via Pyro to the container.
        Read the result information and parse them.

        Parameters
        ----------
        configuration : CS.Configuration, Dict
        fidelity : CS.Configuration, Dict, None
        rng : np.random.RandomState, int, None
        kwargs : Dict

        Returns
        -------
        Dict
        """

        c_str = self._parse_configuration(configuration)
        f_str = self._parse_fidelities(fidelity)
        kwargs_str = self._parse_kwargs(rng=rng, **kwargs)

        json_str = self.benchmark.objective_function_test(c_str, f_str, kwargs_str)
        return json.loads(json_str, cls=BenchmarkDecoder)

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the configuration space object from the benchmark.

        Parameters
        ----------
        seed : int, None
            seed for the configuration space object. If None:  a random seed will be used.

        Returns
        -------
            CS.ConfigurationSpace
        """
        seed_dict = json.dumps({'seed': seed}, indent=None, cls=BenchmarkEncoder)
        logger.debug(f'Client: seed_dict {seed_dict}')
        json_str = self.benchmark.get_configuration_space(seed_dict)

        config_space = csjson.read(json_str)

        if seed is not None:
            config_space.seed(seed)

        return config_space

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the fidelity space as a ConfigurationSpace object from the benchmark.

        Parameters
        ----------
        seed : int, None
            seed for the fidelity space object. If None:  a random seed will be used.

        Returns
        -------
            CS.ConfigurationSpace
        """
        seed_dict = json.dumps({'seed': seed}, indent=None, cls=BenchmarkEncoder)
        logger.debug(f'Client: seed_dict {seed_dict}')
        json_str = self.benchmark.get_fidelity_space(seed_dict)

        fs = csjson.read(json_str)

        if seed is not None:
            fs.seed(seed)

        return fs

    def get_meta_information(self) -> Dict:
        """ Return the information about the benchmark. """
        json_str = self.benchmark.get_meta_information()
        return json.loads(json_str, cls=BenchmarkDecoder)

    def _shutdown(self):
        """ Shutdown benchmark and stop container"""
        try:
            self.benchmark.shutdown()
        except (TypeError, ConnectionRefusedError, Pyro4.errors.CommunicationError, Pyro4.errors.ConnectionClosedError):
            pass

        # If the container is already closed, we dont want a error message here (-> DEVNULL)
        subprocess.run(f'singularity instance stop {self.socket_id}'.split(), check=False, stdout=subprocess.DEVNULL)

        if (self.config.socket_dir / f'{self.socket_id}_unix.sock').exists():
            (self.config.socket_dir / f'{self.socket_id}_unix.sock').unlink()
        # self.benchmark._pyroRelease()
        logger.info('Benchmark is successfully shut down.')

    def __call__(self, configuration: Dict, **kwargs) -> Dict:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    def __del__(self):
        if not self.proxy_only:
            self._shutdown()
        else:
            self.benchmark._pyroRelease()

    @staticmethod
    def _id_generator() -> str:
        """ Helper function: Creates unique socket ids for the benchmark server """
        return str(uuid1())
