#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the server-side for using benchmarks with containers

BenchmarkServer defines the server side for the communication between the
container (benchmark) and the client.
It starts the Pyro4 server and awaits commands from the client. Make sure
that all payloads are json-serializable.
"""

import argparse
import enum
import json
import logging
import os

import ConfigSpace as CS
import Pyro4
import numpy as np
from ConfigSpace.read_and_write import json as csjson

from hpolib.config import HPOlibConfig

logger = logging.getLogger('BenchmarkServer')


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
    def __init__(self, socket_id):
        self.pyroRunning = True
        config = HPOlibConfig()
        self.benchmark = None

        self.socket_id = socket_id
        socket_path = config.socket_dir / (self.socket_id + "_unix.sock")
        if socket_path.exists():
            os.remove(socket_path)
        self.daemon = Pyro4.Daemon(unixsocket=str(socket_path))

        _ = self.daemon.register(self, self.socket_id + ".unixsock")

        # start the event loop of the server to wait for calls
        self.daemon.requestLoop(loopCondition=lambda: self.pyroRunning)

    def init_benchmark(self, kwargs_str):
        try:
            if kwargs_str != "{}":
                kwargs = json.loads(kwargs_str)
                if 'rng' in kwargs and type(kwargs['rng']) == list:
                    (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng']
                    rnd1 = [np.uint32(number) for number in rnd1]
                    kwargs['rng'] = np.random.set_state((rnd0, rnd1, rnd2, rnd3, rnd4))
                    logger.debug('rng works!!')
                self.benchmark = Benchmark(**kwargs)
            else:
                self.benchmark = Benchmark()
        except Exception as e:
            print(e)
            logger.error(e)

    def get_configuration_space(self):
        result = self.benchmark.get_configuration_space()
        return csjson.write(result, indent=None)

    def objective_function_list(self, x_str, kwargs_str):
        x = json.loads(x_str)
        result = self.benchmark.objective_function(x, **json.loads(kwargs_str))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function(self, c_str, cs_str, kwargs_str):
        c_dict = json.loads(c_str)
        cs = csjson.read(cs_str)
        configuration = CS.Configuration(cs, c_dict)
        result = self.benchmark.objective_function(configuration, **json.loads(kwargs_str))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test_list(self, x_str, kwargs_str):
        x = json.loads(x_str)
        result = self.benchmark.objective_function_test(x, **json.loads(kwargs_str))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test(self, c_str, cs_str, kwargs_str):
        c_dict = json.loads(c_str)
        cs = csjson.read(cs_str)
        configuration = CS.Configuration(cs, c_dict)
        result = self.benchmark.objective_function_test(configuration, **json.loads(kwargs_str))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def test(self, args_str, kwargs_str):
        result = self.benchmark.test(*json.loads(args_str), **json.loads(kwargs_str))
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def get_meta_information(self):
        return json.dumps(self.benchmark.get_meta_information(), indent=None)

    @Pyro4.oneway   # in case call returns much later than daemon.shutdown
    def shutdown(self):
        logger.debug('shutting down...')
        Pyro4.config.COMMTIMEOUT = 0.5
        self.pyroRunning = False
        self.daemon.shutdown()


if __name__ == "__main__":
    Pyro4.config.REQUIRE_EXPOSE = False

    parser = argparse.ArgumentParser(prog='server_abstract_benchmark.py',
                                     description='HPOlib3 Container Server',
                                     usage='%(prog)s <importBase> <benchmark> <socket_id>')
    parser.add_argument('importBase', type=str,
                        help='Relative path to benchmark file in hpolib/benchmarks, e.g. ml.xgboost_benchmark')
    parser.add_argument('benchmark', type=str,
                        help='Classname of the benchmark, e.g. XGBoostBenchmark')
    parser.add_argument('socket_id', type=str,
                        help='socket_id for pyro-server')

    args = parser.parse_args()

    exec(f"from hpolib.benchmarks.{args.importBase} import {args.benchmark} as Benchmark")
    bp = BenchmarkServer(args.socket_id)
