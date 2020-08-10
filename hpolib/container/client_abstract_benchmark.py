#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the client-side for using benchmarks with containers

AbstractBenchmarkClient defines the client side for the communication between
the containers and the client.
It is used to download (if not already) and start containers in the background.

To reduce download traffic, firstly, it checks if the container is already
downloaded. The container source as well as the path, where it should be stored,
are defined in the ~/.hpolibrc - file.

The name of the container (``container_name``) is defined either in its belonging
container-benchmark definition. (hpolib/container/<type>/<name> or via ``container_name``.
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
from typing import Union, Dict, List
from uuid import uuid1

import ConfigSpace as CS
import Pyro4
import Pyro4.util
import numpy as np
from ConfigSpace.read_and_write import json as csjson
from oslo_concurrency import lockutils

import hpolib.config

# Read in the verbosity level from the environment variable HPOLIB_DEBUG
log_level_str = os.environ.get('HPOLIB_DEBUG', 'false')
log_level = logging.DEBUG if log_level_str == 'true' else logging.INFO

root = logging.getLogger()
root.setLevel(level=log_level)

logger = logging.getLogger("BenchmarkClient")
logger.setLevel(level=log_level)

# This option improves the quality of stacktraces if a container crashes
sys.excepthook = Pyro4.util.excepthook


class AbstractBenchmarkClient(metaclass=abc.ABCMeta):
    """ Base Class for the containerized benchmarks.

    Attributes
    ----------
    socket_id : str
    """
    def __init__(self, benchmark_name: str, container_name: str, container_source: Optional[str] = None,
                 gpu: Optional[bool] = False, rng: Union[np.random.RandomState, int, None] = None, **kwargs):

        self.socket_id = self._id_generator()

        if rng is not None:
            kwargs['rng'] = rng

        self._setup(benchmark_name, container_name, container_source, gpu, **kwargs)

    def _setup(self, benchmark_name: str, container_name: str, container_source: Optional[str] = None,
               gpu: bool = False, **kwargs):
        """ Initialization of the benchmark using container.

        This setup function downloads the container from a defined source. The source is defined either in the
        .hpolibrc or in the its benchmark definition (hpolib/container/benchmarks/<type>/<name>). If an container
        is already locally available, the local container is used. Then, the container is started and a connection
        between the container and the client is established.

        Parameters
        ----------
        benchmark_name: str
            Class name of the benchmark to use. For example XGBoostBenchmark. This value is defined in the benchmark
            definition (hpolib/container/benchmarks/<type>/<name>)
        container_source : Optional[str]
            Path to the container. Either local path or url to a hosting
            platform, e.g. singularity hub.
        container_name : str
            name of the container. E.g. xgboost_benchmark. Specifying different container could be
            useful to have multiple container for the same benchmark, if a tool like auto-sklearn is updated to a newer
            version, and you want to compare its performance across its versions.
            Also, a container can contain multiple benchmarks. Therefore, we have to define for each benchmark the
            corresponding container name.
        gpu : bool
            If True, the container has access to the local cuda-drivers.
            (Not tested)
        """
        # Create unique ID
        self.config = hpolib.config.config_file

        # We can point to a different container source. See below.
        container_source = container_source or self.config.container_source
        container_dir = Path(self.config.container_dir)

        logger.debug(f'Use benchmark {benchmark_name} from container {container_source}/{container_name}. \n'
                     f'And container directory {self.config.container_dir}')

        # Pull the container from the singularity hub if the container is hosted online. If the container is stored
        # locally (e.g.for development) do not pull it.
        if container_source is not None \
                and any((s in container_source for s in ['shub', 'library', 'docker', 'oras', 'http'])):

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
            # We limit the number of lock file by having at most one lock file per benchmark and storing them in the
            # temp folder, so that they are automatically deleted after reboot.
            @lockutils.synchronized('not_thread_process_safe', external=True,
                                    lock_path=f'{self.config.socket_dir}/lock_{container_name}', delay=5)
            def download_container(container_dir, container_name, container_source):
                if not (container_dir / container_name).exists():
                    logger.debug('Going to pull the container from an online source.')

                    container_dir.mkdir(parents=True, exist_ok=True)
                    cmd = f"singularity pull --dir {self.config.container_dir} " \
                          f"--name {container_name} {container_source}/{container_name.lower()}"
                    logger.debug(cmd)
                    subprocess.run(cmd, shell=True)
                    time.sleep(1)
                else:
                    logger.debug('Skipping downloading the container. It is already downloaded.')

            download_container(container_dir, container_name, container_source)
        else:
            logger.debug(f'Looking on the local filesystem for the container file, since container source was '
                         f'either \'None\' or not a known address. Image Source: {container_source}')

            # Make sure that the container can be found locally.
            container_dir = Path(container_source)
            assert (container_dir / container_name).exists(), f'Local container not found in ' \
                                                              f'{container_dir / container_name}'
            logger.debug('Image found on the local file system.')

        bind_options = f'--bind /var/lib/,{self.config.global_data_dir}:/var/lib/'
        if self.config.global_data_dir != self.config.data_dir:
            bind_options += f',{self.config.data_dir}:/var/lib/'
        bind_options += ' '

        gpu_opt = '--nv ' if gpu else ''  # Option for enabling GPU support
        container_options = f'{container_dir / container_name}'

        log_str = f'SINGULARITYENV_HPOLIB_DEBUG={log_level_str}'
        cmd = f'{log_str} singularity instance start {bind_options}{gpu_opt}{container_options} {self.socket_id}'
        logger.debug(cmd)

        MAX_TRIES = 5
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
            else:
                logger.debug(f'Could not start instance: Try {num_tries + 1}|{MAX_TRIES}')
                if num_tries + 1 == MAX_TRIES:
                    raise SystemError(f'Could not start a instance of the benchmark. Retried {MAX_TRIES:d} times'
                                      f'\nStdout: {output} \nStderr: {err}')
            st = np.random.randint(1, 60)
            logger.critical(f'[{num_tries + 1}/{MAX_TRIES}] Could not start instance, sleeping for {st} seconds')
            time.sleep(st)

        # Give each instance a little bit time to start
        time.sleep(1)

        cmd = f'singularity run {gpu_opt}instance://{self.socket_id} {benchmark_name} {self.socket_id}'
        logger.debug(cmd)
        subprocess.Popen(cmd.split())

        Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        self.uri = f'PYRO:{self.socket_id}.unixsock@./u:' \
                   f'{self.config.socket_dir}/{self.socket_id}_unix.sock'
        self.benchmark = Pyro4.Proxy(self.uri)

        # Handle rng and other optional benchmark arguments
        if 'rng' in kwargs and isinstance(kwargs['rng'], np.random.RandomState):
            (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng'].get_state()
            rnd1 = [int(number) for number in rnd1]
            kwargs['rng'] = (rnd0, rnd1, rnd2, rnd3, rnd4)
        kwargs_str = json.dumps(kwargs)

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
                else:
                    logger.debug('Waiting time exceeded. To increase, adjust config option pyro_connect_max_wait.')
                    raise
            break
        logger.debug('Connected to container')

    def _parse_kwargs(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """ Helper function to parse the named keyword arguments to json str. """
        if rng is not None:
            rng = self._cast_random_state_to_int(rng)
            kwargs['rng'] = rng
        kwargs_str = json.dumps(kwargs, indent=None)
        return kwargs_str

    def _parse_fidelities(self, fidelity: Union[CS.Configuration, Dict, None] = None):
        if fidelity is None:
            fidelity = {}
        elif isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()
        elif isinstance(fidelity, dict):
            fidelity = fidelity
        else:
            raise ValueError(f'Type of fidelity not understood: {type(fidelity)}')
        f_str = json.dumps(fidelity, indent=None)
        return f_str

    def objective_function(self, configuration: Union[np.ndarray, List, CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Run a given configuration for a given fidelity on the containerized benchmark.

        Convert the given parameters to strings and send them via Pyro to the container.
        Read the result information and parse them.

        Parameters
        ----------
        configuration : np.ndarray, List, CS.Configuration, Dict
        fidelity : CS.Configuration, Dict, None
        rng : np.random.RandomState, int, None
        kwargs : Dict

        Returns
        -------
        Dict
        """

        kwargs_str = self._parse_kwargs(rng, **kwargs)
        f_str = self._parse_fidelities(fidelity)

        if isinstance(configuration, np.ndarray):
            configuration = configuration.tolist()

        if isinstance(configuration, list):
            c_str = json.dumps(configuration, indent=None)
            json_str = self.benchmark.objective_function_list(c_str, f_str, kwargs_str)
            return json.loads(json_str)
        elif isinstance(configuration, CS.Configuration):
            c_str = json.dumps(configuration.get_dictionary(), indent=None)
        elif isinstance(configuration, dict):
            c_str = json.dumps(configuration, indent=None)
        else:
            raise ValueError(f'Type of config not understood: {type(configuration)}')

        json_str = self.benchmark.objective_function(c_str, f_str, kwargs_str)
        return json.loads(json_str)

    def objective_function_test(self, configuration: Union[np.ndarray, List, CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Run a given configuration for a given fidelity on the test function of the containerized  benchmark.

        Convert the given parameters to strings and send them via Pyro to the container.
        Read the result information and parse them.

        Parameters
        ----------
        configuration : np.ndarray, List, CS.Configuration, Dict
        fidelity : CS.Configuration, Dict, None
        rng : np.random.RandomState, int, None
        kwargs : Dict

        Returns
        -------
        Dict
        """

        kwargs_str = self._parse_kwargs(rng=rng, **kwargs)
        f_str = self._parse_fidelities(fidelity)

        if isinstance(configuration, np.ndarray):
            configuration = configuration.tolist()

        if isinstance(configuration, list):
            c_str = json.dumps(configuration, indent=None)
            json_str = self.benchmark.objective_function_test_list(c_str, f_str, kwargs_str)
            return json.loads(json_str)
        elif isinstance(configuration, CS.Configuration):
            c_str = json.dumps(configuration.get_dictionary(), indent=None)
        elif isinstance(configuration, dict):
            c_str = json.dumps(configuration, indent=None)

        else:
            raise ValueError(f'Type of config not understood: {type(configuration)}')

        json_str = self.benchmark.objective_function_test(c_str, f_str, kwargs_str)
        return json.loads(json_str)

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
        seed_dict = {}
        if seed is not None:
            seed_dict['seed'] = seed
        seed_dict = json.dumps(seed_dict, indent=None)
        logger.debug(f'Client: seed_dict {seed_dict}')
        json_str = self.benchmark.get_configuration_space(seed_dict)
        return csjson.read(json_str)

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
        seed_dict = {}
        if seed is not None:
            seed_dict['seed'] = seed
        seed_dict = json.dumps(seed_dict, indent=None)
        logger.debug(f'Client: seed_dict {seed_dict}')
        json_str = self.benchmark.get_fidelity_space(seed_dict)
        return csjson.read(json_str)

    def get_meta_information(self) -> Dict:
        """ Return the information about the benchmark. """
        json_str = self.benchmark.get_meta_information()
        return json.loads(json_str)

    def __call__(self, configuration: Dict, **kwargs) -> Dict:
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    def __del__(self):
        self.benchmark.shutdown()
        subprocess.run(f'singularity instance stop {self.socket_id}'.split())
        if (self.config.socket_dir / f'{self.socket_id}_unix.sock').exists():
            (self.config.socket_dir / f'{self.socket_id}_unix.sock').unlink()
        # self.benchmark._pyroRelease()

    @staticmethod
    def _id_generator() -> str:
        """ Helper function: Creates unique socket ids for the benchmark server """
        return str(uuid1())

    @staticmethod
    def _cast_random_state_to_int(rng: [int, np.random.RandomState]) -> int:
        if isinstance(rng, np.random.RandomState):
            rng = rng.randint(0, 100000)
        return rng
