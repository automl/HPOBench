#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the client-side for using benchmarks with containers

AbstractBenchmarkClient defines the client side for the communication between 
the containers and the client. 
It is used to download (if not already) and start containers in the background.

To reduce download traffic, firstly, it checks if the image is already downloaded.
The container source as well the path, where it should be stored, are defined in the
~/.hpolibrc - file.

The name of the container (`benchmark_name`) is defined either in its belonging container-benchmark definition.
(hpolib/container/<type>/<name> or via `ingName`.
"""

import abc
import json
import numpy
import os
import random
import string
import subprocess
import time
import Pyro4
from ConfigSpace.read_and_write import json as csjson
import hpolib.config
from pathlib import Path
from typing import Optional
import logging


class AbstractBenchmarkClient(metaclass=abc.ABCMeta):
    """ Base Class for the containerized benchmarks.

    Attributes
    ----------
    logger : logging.Logger

    """
    def __init__(self):
        self.logger = logging.getLogger("BenchmarkClient")
        self.socketId = self._id_generator()

    def _setup(self, gpu: bool = False, img_name: Optional[str] = None, img_source: Optional[str] = None, **kwargs):
        """ Initialization of the benchmark using container.

        This setup function downloads the container from a defined source. The source is defined either in the .hpolibrc
        or in the its benchmark definition (hpolib/container/benchmarks/..). If a image is already in the local
        available, the local image is used.
        Then, the container is started and a connection between the container and the client is established.

        Parameters
        ----------
        gpu : bool
            If True, the container has access to the local Cuda-drivers. (Not tested)
        img_name : Optional[str]
            name of the image. E.g. XGBoostOnMnist. The local container has to have the same name.
        img_source : Optional[str]
            Path to the image. Either local path or link to singularity hub, etc
        """
        # Create unique ID
        self.config = hpolib.config.config_file

        # Default image name is benchmark name. img_name can be specified to point to another container.
        img_name = img_name or self.benchmark_name

        # Same for the image's source.
        img_source = img_source or self.config.image_source
        self.logger.debug(f'Image {img_name} in {self.config.image_dir} from {img_source}')

        img_dir = Path(self.config.image_dir)

        # Pull the image from the singularity hub if the image is hosted online. If the image is stored locally (e.g.
        # for development) do not pull it.
        if img_source is not None \
                and any((s in img_source for s in ['shub', 'library', 'docker', 'oras', 'http'])):

            if not (img_dir / img_name).exists():
                self.logger.debug('Going to pull the image from an online source.')

                img_dir.mkdir(parents=True, exist_ok=True)
                cmd = f"singularity pull --dir {self.config.image_dir} --name {img_name} " \
                      f"{img_source}:{img_name.lower()}"
                self.logger.debug(cmd)
                subprocess.run(cmd, shell=True)
            else:
                self.logger.debug('Skipping downloading the image. It is already downloaded.')
        else:
            self.logger.debug('Looking on the local filesystem for the image file, since image source was '
                              'either \'None\' or not a known address. '
                              f'Image Source: {img_source}')

            # Make sure that the image can be found locally.
            assert (img_dir / img_name).exists(), f'Local image not found in {img_dir / img_name}'
            self.logger.debug('Image found on the local file system.')

        iOptions = str(img_dir / img_name)
        sOptions = f'{self.benchmark_name} {self.socketId}'

        # Option for enabling GPU support
        gpuOpt = '--nv ' if gpu else ''

        cmd = f'singularity instance start --bind /var/lib/ {gpuOpt}{iOptions} {self.socketId}'
        self.logger.debug(cmd)
        subprocess.run(cmd, shell=True)

        cmd = f'singularity run {gpuOpt}instance://{self.socketId} {sOptions}'
        self.logger.debug(cmd)
        subprocess.Popen(cmd, shell=True)

        Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        self.uri = f'PYRO:{self.socketId}.unixsock@./u:{self.config.socket_dir}/{self.socketId}_unix.sock'
        self.b = Pyro4.Proxy(self.uri)

        # Handle rng and other optional benchmark arguments
        if 'rng' in kwargs and type(kwargs['rng']) == numpy.random.RandomState:
            (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng'].get_state()
            rnd1 = [int(number) for number in rnd1]
            kwargs['rng'] = (rnd0, rnd1, rnd2, rnd3, rnd4)
        kwargsStr = json.dumps(kwargs)

        # Try to connect to server calling benchmark constructor via RPC.
        # There exist a time limit
        self.logger.debug('Check connection to container and init benchmark')
        wait = 0
        while True:
            try:
                self.b.initBenchmark(kwargsStr)
            except Pyro4.core.errors.CommunicationError:
                self.logger.debug('Still waiting')
                time.sleep(5)
                wait += 5
                if wait < self.config.pyro_connect_max_wait:
                    continue
                else:
                    self.logger.debug('Waiting time exceeded. To high it up, adjust config '
                                      'option pyro_connect_max_wait.')
                    raise
            break
        self.logger.debug('Connected to container')

    def objective_function(self, x, **kwargs):
        # Create the arguments as Str
        if type(x) is list:
            xString = json.dumps(x, indent=None)
            jsonStr = self.b.objective_function_list(xString, json.dumps(kwargs))
            return json.loads(jsonStr)
        else:
            # Create the arguments as Str
            cString = json.dumps(x.get_dictionary(), indent=None)
            csString = csjson.write(x.configuration_space, indent=None)
            jsonStr = self.b.objective_function(cString, csString, json.dumps(kwargs))
            return json.loads(jsonStr)

    def objective_function_test(self, x, **kwargs):
        # Create the arguments as Str
        if type(x) is list:
            xString = json.dumps(x, indent=None)
            jsonStr = self.b.objective_function_test_list(xString, json.dumps(kwargs))
            return json.loads(jsonStr)
        else:
            # Create the arguments as Str
            cString = json.dumps(x.get_dictionary(), indent=None)
            csString = csjson.write(x.configuration_space, indent=None)
            jsonStr = self.b.objective_function_test(cString, csString, json.dumps(kwargs))
            return json.loads(jsonStr)

    def test(self, *args, **kwargs):
        result = self.b.test(json.dumps(args), json.dumps(kwargs))
        return json.loads(result)

    def get_configuration_space(self):
        jsonStr = self.b.get_configuration_space()
        return csjson.read(jsonStr)

    def get_meta_information(self):
        jsonStr = self.b.get_meta_information()
        return json.loads(jsonStr)

    def __call__(self, configuration, **kwargs):
        """ Provides interface to use, e.g., SciPy optimizers """
        return self.objective_function(configuration, **kwargs)['function_value']

    def _id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def __del__(self):
        Pyro4.config.COMMTIMEOUT = 1
        self.b.shutdown()
        subprocess.run(f'singularity instance stop {self.socketID}', shell=True)
        os.remove(self.config.socket_dir + self.socketId + '_unix.sock')
