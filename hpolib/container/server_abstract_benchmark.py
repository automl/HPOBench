#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the server-side for using benchmarks with containers

BenchmarkServer defines the server side for the communication between the container (benchmark) and the client.
It starts the Pyro4 server and awaits commands from the client. Make sure that all payloads are json-serializable.
"""

import enum
import numpy as np
import os
import argparse
import json

import Pyro4
import logging
import ConfigSpace as CS
from ConfigSpace.read_and_write import json as csjson
from hpolib.config import HPOlibConfig


class BenchmarkEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, enum.Enum):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class BenchmarkServer:
    def __init__(self, socketId):
        self.pyroRunning = True
        config = HPOlibConfig()
        self.logger = logging.getLogger('BenchmarkServer')
        self.b = None

        self.socketId = socketId
        socketPath = config.socket_dir / (self.socketId + "_unix.sock")
        if socketPath.exists():
            os.remove(socketPath)
        self.daemon = Pyro4.Daemon(unixsocket=str(socketPath))

        uri = self.daemon.register(self, self.socketId + ".unixsock")

        # start the event loop of the server to wait for calls
        self.daemon.requestLoop(loopCondition=lambda: self.pyroRunning)

    def initBenchmark(self, kwargsStr):
        try:
            if kwargsStr != "{}":
                kwargs = json.loads(kwargsStr)
                if 'rng' in kwargs and type(kwargs['rng']) == list:
                    (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng']
                    rnd1 = [np.uint32(number) for number in rnd1]
                    kwargs['rng'] = np.random.set_state((rnd0, rnd1, rnd2, rnd3, rnd4))
                    self.logger.debug('rng works!!')
                self.b = Benchmark(**kwargs)
            else:
                self.b = Benchmark()
        except Exception as e:
            print(e)
            self.logger.error(e)

    def get_configuration_space(self):
        result = self.b.get_configuration_space()
        return csjson.write(result, indent=None)

    def objective_function_list(self, xString, kwargsStr):
        x = json.loads(xString)
        result = self.b.objective_function(x, **json.loads(kwargsStr))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function(self, cString, csString, kwargsStr):
        cDict = json.loads(cString)
        cs = csjson.read(csString)
        configuration = CS.Configuration(cs, cDict)
        result = self.b.objective_function(configuration, **json.loads(kwargsStr))
        # Handle SMAC status
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test_list(self, xString, kwargsStr):
        x = json.loads(xString)
        result = self.b.objective_function_test(x, **json.loads(kwargsStr))
        # Handle SMAC runhistory
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test(self, cString, csString, kwargsStr):
        cDict = json.loads(cString)
        cs = csjson.read(csString)
        configuration = CS.Configuration(cs, cDict)
        result = self.b.objective_function_test(configuration, **json.loads(kwargsStr))
        # Handle SMAC runhistory
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def test(self, argsStr, kwargsStr):
        result = self.b.test(*json.loads(argsStr), **json.loads(kwargsStr))
        # Handle SMAC runhistory
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def get_meta_information(self):
        return json.dumps(self.b.get_meta_information(), indent=None)

    @Pyro4.oneway   # in case call returns much later than daemon.shutdown
    def shutdown(self):
        self.logger.debug('shutting down...')
        Pyro4.config.COMMTIMEOUT = 0.5
        self.pyroRunning = False
        self.daemon.shutdown()


if __name__ == "__main__":
    Pyro4.config.REQUIRE_EXPOSE = False

    parser = argparse.ArgumentParser(prog='server_abstract_benchmark.py',
                                     description='HPOlib3 Container Server',
                                     usage='%(prog)s <importBase> <benchmark> <socketId>')
    parser.add_argument('importBase', type=str,
                        help='Relative path to benchmark file in hpolib/benchmarks, e.g. ml.xgboost_benchmark')
    parser.add_argument('benchmark', type=str,
                        help='Classname of the benchmark, e.g. XGBoostOnMnist')
    parser.add_argument('socketId', type=str,
                        help='SocketId for pyro-server')

    args = parser.parse_args()
    # module = importlib.import_module(f'hpolib.benchmarks.{importBase}')
    # Benchmark = getattr(module, benchmark)
    exec(f"from hpolib.benchmarks.{args.importBase} import {args.benchmark} as Benchmark")
    bp = BenchmarkServer(args.socketId)
