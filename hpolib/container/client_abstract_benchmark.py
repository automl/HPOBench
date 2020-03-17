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

import abc
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
from uuid import uuid1

import Pyro4
import numpy
from ConfigSpace.read_and_write import json as csjson
from ConfigSpace import Configuration

import hpolib.config

logger = logging.getLogger("BenchmarkClient")


class AbstractBenchmarkClient(metaclass=abc.ABCMeta):
    """ Base Class for the containerized benchmarks.

    Attributes
    ----------
    socket_id : str

    """
    def __init__(self):
        self.socket_id = self._id_generator()

    def _setup(self, benchmark_name: str, container_source: Optional[str] = None, container_name: Optional[str] = None,
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
            definition (hpolib/container/benchmarks/<type>/<name>
        container_source : Optional[str]
            Path to the container. Either local path or url to a hosting
            platform, e.g. singularity hub.
        container_name : Optional[str]
            name of the container. E.g. xgboost_benchmark. Specifying different container could be
            useful to have multiple container for the same benchmark, if a tool like auto-sklearn is updated to a newer
            version, and you want to compare its performance across its versions.
        gpu : bool
            If True, the container has access to the local cuda-drivers.
            (Not tested)
        """
        # Create unique ID
        self.config = hpolib.config.config_file

        # Default container name is benchmark name. container_name can be specified to point to another container.
        container_name = container_name or benchmark_name

        # Same for the container's source.
        container_source = container_source or self.config.container_source
        container_dir = Path(self.config.container_dir)

        logger.debug(f'Use benchmark {benchmark_name} from container {container_source}/{container_name}. \n'
                     f'And container directory {self.config.container_dir}')

        # Pull the container from the singularity hub if the container is hosted online. If the container is stored
        # locally (e.g.for development) do not pull it.
        if container_source is not None \
                and any((s in container_source for s in ['shub', 'library', 'docker', 'oras', 'http'])):

            if not (container_dir / container_name).exists():
                logger.debug('Going to pull the container from an online source.')

                container_dir.mkdir(parents=True, exist_ok=True)
                cmd = f"singularity pull --dir {self.config.container_dir} " \
                      f"--name {container_name} {container_source}/{container_name.lower()}"
                logger.debug(cmd)
                subprocess.run(cmd, shell=True)
            else:
                logger.debug('Skipping downloading the container. It is already downloaded.')
        else:
            logger.debug('Looking on the local filesystem for the container file, since container source was '
                         'either \'None\' or not a known address. Image Source: {container_source}')

            # Make sure that the container can be found locally.
            container_dir = Path(container_source)
            assert (container_dir / container_name).exists(), f'Local container not found in ' \
                                                              f'{container_dir / container_name}'
            logger.debug('Image found on the local file system.')

        bind_options = f'--bind /var/lib/,{self.config.global_data_dir}:/var/lib/,{self.config.data_dir}:/var/lib/ '
        gpu_opt = '--nv ' if gpu else ''  # Option for enabling GPU support
        container_options = f'{container_dir / container_name}'

        cmd = f'singularity instance start {bind_options}{gpu_opt}{container_options} {self.socket_id}'
        logger.debug(cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'singularity run {gpu_opt}instance://{self.socket_id} {benchmark_name} {self.socket_id}'
        logger.debug(cmd)
        subprocess.Popen(cmd, shell=True)

        Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        self.uri = f'PYRO:{self.socket_id}.unixsock@./u:' \
                   f'{self.config.socket_dir}/{self.socket_id}_unix.sock'
        self.benchmark = Pyro4.Proxy(self.uri)

        # Handle rng and other optional benchmark arguments
        if 'rng' in kwargs and isinstance(kwargs['rng'], numpy.random.RandomState):
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

    def objective_function(self, x, **kwargs):
        if isinstance(x, list):
            x_str = json.dumps(x, indent=None)
            json_str = self.benchmark.objective_function_list(x_str, json.dumps(kwargs))
            return json.loads(json_str)
        elif isinstance(x, Configuration):
            c_str = json.dumps(x.get_dictionary(), indent=None)
            cs_str = csjson.write(x.configuration_space, indent=None)
            json_str = self.benchmark.objective_function(c_str, cs_str, json.dumps(kwargs))
            return json.loads(json_str)
        else:
            raise ValueError(f'Type of config not understood: {type(x)}')

    def objective_function_test(self, x, **kwargs):
        if isinstance(x, list):
            x_str = json.dumps(x, indent=None)
            json_str = self.benchmark.objective_function_test_list(x_str, json.dumps(kwargs))
            return json.loads(json_str)
        elif isinstance(x, Configuration):
            c_str = json.dumps(x.get_dictionary(), indent=None)
            cs_str = csjson.write(x.configuration_space, indent=None)
            json_str = self.benchmark.objective_function_test(c_str, cs_str, json.dumps(kwargs))
            return json.loads(json_str)
        else:
            raise ValueError(f'Type of config not understood: {type(x)}')

    def test(self, *args, **kwargs):
        result = self.benchmark.test(json.dumps(args), json.dumps(kwargs))
        return json.loads(result)

    def get_configuration_space(self):
        json_str = self.benchmark.get_configuration_space()
        return csjson.read(json_str)

    def get_meta_information(self):
        json_str = self.benchmark.get_meta_information()
        return json.loads(json_str)

    def __call__(self, configuration, **kwargs):
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    def __del__(self):
        Pyro4.config.COMMTIMEOUT = 1
        self.benchmark.shutdown()
        subprocess.run(f'singularity instance stop {self.socket_id}', shell=True)
        os.remove(str(self.config.socket_dir / f'{self.socket_id}_unix.sock'))

    def _id_generator(self):
        return str(uuid1())
