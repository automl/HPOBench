#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the client-side for using benchmarks with containers

AbstractBenchmarkClient defines the client side for the communication between 
the containers and the client. 
It is used to download (if not already) and start containers in the background.

To reduce download traffic, firstly, it checks if the image is already downloaded.
The container source as well the path, where it should be stored, are defined in the
~/.hpolibrc - file.

The name of the container (`bName`) is defined either in its belonging container-benchmark definition.
(hpolib/container/<type>/<name> or via `ingName`.


@author: Stefan Staeglich, Philipp Mueller
@maintainer: Philipp Mueller (muelleph@informatik.uni-freiburg.de)
"""

import abc
import json
import numpy
import os
import random
import signal
import string
import subprocess
import time
import Pyro4
from ConfigSpace.read_and_write import json as csjson
import hpolib.config
from pathlib import Path


class AbstractBenchmarkClient(metaclass=abc.ABCMeta):
    def _setup(self, gpu=False, imgName=None, **kwargs):
        # Create unique ID
        self.socketId = self._id_generator()
        self.config = hpolib.config._config

        # Default image name is benchmark name. imgName can be specified to point to another container.
        imgName = imgName or self.bName
        self.config.logger.debug(f'Image {imgName} in {self.config.image_dir} from {self.config.image_source}')

        img_dir = Path(self.config.image_dir)

        # Pull the image from the singularity hub if the image is hosted online. If the image is stored locally (e.g.
        # for development) do not pull it.
        if self.config.image_source is not None \
                and any((s in self.config.image_source for s in ['shub', 'library', 'docker', 'oras', 'http'])):

            if not (img_dir / imgName).exists():
                self.config.logger.debug('Going to pull the image from an online source.')

                img_dir.mkdir(parents=True, exist_ok=True)
                cmd = f"singularity pull --dir {self.config.image_dir} --name {imgName} " \
                      f"{self.config.image_source}:{imgName.lower()}"
                self.config.logger.debug(cmd)
                subprocess.run(cmd, shell=True)
            else:
                self.config.logger.debug('Skipping downloading the image. It is already downloaded.')
        else:
            self.config.logger.debug('Looking on the local filesystem for the image file, since image source was '
                                     'either \'None\' or not a known address. '
                                     f'Image Source: {self.config.image_source}')

            # Make sure that the image can be found locally.
            assert (img_dir / imgName).exists(), f'Local image not found in {img_dir / imgName}'
            self.config.logger.debug('Image found on the local file system.')

        iOptions = str(img_dir / imgName)
        sOptions = self.bName + " " + self.socketId

        # Option for enabling GPU support
        gpuOpt = "--nv " if gpu else ""

        # By default use named singularity instances.
        # There exist a config option to disable this behaviour
        if self.config.singularity_use_instances:
            cmd = f"singularity instance start --bind /var/lib/ {gpuOpt}{iOptions} {self.socketId}"
            self.config.logger.debug(cmd)
            subprocess.run(cmd, shell=True)

            cmd = f"singularity run {gpuOpt}instance://{self.socketId} {sOptions}"
            self.config.logger.debug(cmd)
            subprocess.Popen(cmd, shell=True)
        else:
            self.sProcess = subprocess.Popen(f"singularity run "
                                             f"{gpuOpt}{iOptions} {sOptions}",
                                             shell=True)

        Pyro4.config.REQUIRE_EXPOSE = False
        # Generate Pyro 4 URI for connecting to client
        self.uri = f"PYRO:{self.socketId}.unixsock@./u:{self.config.socket_dir}/{self.socketId}_unix.sock"
        self.b = Pyro4.Proxy(self.uri)

        # Handle rng and other optional benchmark arguments
        if 'rng' in kwargs and type(kwargs['rng']) == numpy.random.RandomState:
            (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng'].get_state()
            rnd1 = [int(number) for number in rnd1]
            kwargs['rng'] = (rnd0, rnd1, rnd2, rnd3, rnd4)
        kwargsStr = json.dumps(kwargs)

        # Try to connect to server calling benchmark constructor via RPC.
        # There exist a time limit
        self.config.logger.debug("Check connection to container and init benchmark")
        wait = 0
        while True:
            try:
                self.b.initBenchmark(kwargsStr)
            except Pyro4.errors.CommunicationError:
                self.config.logger.debug("Still waiting")
                time.sleep(5)
                wait += 5
                if wait < self.config.pyro_connect_max_wait:
                    continue
                else:
                    self.config.logger.debug("Waiting time exceeded. To high it up, "
                                             "adjust config option pyro_connect_max_wait.")
                    raise
            break
        self.config.logger.debug("Connected to container")

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
        return(self.objective_function(configuration, **kwargs)['function_value'])

    def _id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def __del__(self):
        Pyro4.config.COMMTIMEOUT = 1
        self.b.shutdown()
        if self.config.singularity_use_instances:
            subprocess.run("singularity instance stop %s" % (self.socketId), shell=True)
        else:
            os.killpg(os.getpgid(self.sProcess.pid), signal.SIGTERM)
            self.sProcess.terminate()
        os.remove(self.config.socket_dir + self.socketId + "_unix.sock")
