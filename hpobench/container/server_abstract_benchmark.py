#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Defines the server-side for using benchmarks with containers

BenchmarkServer defines the server side for the communication between the
container (benchmark) and the client.
It starts the Pyro4 server and awaits commands from the client. Make sure
that all payloads are json-serializable.
"""

import argparse
import json
import logging
import os

import Pyro4
from ConfigSpace.read_and_write import json as csjson

from hpobench.config import HPOBenchConfig
from hpobench.util.container_utils import BenchmarkEncoder, BenchmarkDecoder

# Read in the verbosity level from the environment variable HPOBENCH_DEBUG
log_level_str = os.environ.get('HPOBENCH_DEBUG', 'false')
LOG_LEVEL = logging.DEBUG if log_level_str == 'true' else logging.INFO

root = logging.getLogger("hpobench")
root.setLevel(LOG_LEVEL)

logger = logging.getLogger('BenchmarkServer')


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class BenchmarkServer:
    def __init__(self, socket_id):
        self.pyro_running = True
        config = HPOBenchConfig()
        self.benchmark = None

        self.socket_id = socket_id
        socket_path = config.socket_dir / (self.socket_id + "_unix.sock")
        logger.debug(f'Socket Path: {socket_path}')
        logger.info(f'Logging level: {logger.level}')

        if socket_path.exists():
            os.remove(socket_path)
        self.daemon = Pyro4.Daemon(unixsocket=str(socket_path))

        _ = self.daemon.register(self, self.socket_id + ".unixsock")

        # start the event loop of the server to wait for calls
        self.daemon.requestLoop(loopCondition=lambda: self.pyro_running)

    def init_benchmark(self, kwargs_str):
        try:
            kwargs = json.loads(kwargs_str, cls=BenchmarkDecoder)
            self.benchmark = Benchmark(**kwargs)  # noqa: F821
            logger.info('Server: Connected Successfully')
        except Exception as e:
            logger.exception(e)

    def get_configuration_space(self, kwargs_str: str) -> str:
        logger.debug(f'Server: get_config_space: kwargs_str: {kwargs_str}')

        kwargs = json.loads(kwargs_str, cls=BenchmarkDecoder)
        seed = kwargs.get('seed', None)

        result = self.benchmark.get_configuration_space(seed=seed)
        logger.debug(f'Server: Configspace: {result}')
        return csjson.write(result, indent=None)

    def get_fidelity_space(self, kwargs_str: str) -> str:
        logger.debug(f'Server: get_fidelity_space: kwargs_str: {kwargs_str}')

        kwargs = json.loads(kwargs_str, cls=BenchmarkDecoder)
        seed = kwargs.get('seed', None)

        result = self.benchmark.get_fidelity_space(seed=seed)
        logger.debug(f'Server: Fidelity Space: {result}')
        return csjson.write(result, indent=None)

    def objective_function(self, c_str: str, f_str: str, kwargs_str: str) -> str:
        logger.debug(f'Server: objective_function: c_str: {c_str} f_str: {f_str} kwargs_str: {kwargs_str}')

        configuration = json.loads(c_str, cls=BenchmarkDecoder)
        fidelity = json.loads(f_str, cls=BenchmarkDecoder)
        kwargs = json.loads(kwargs_str, cls=BenchmarkDecoder)

        result = self.benchmark.objective_function(configuration=configuration, fidelity=fidelity, **kwargs)
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test(self, c_str: str, f_str: str, kwargs_str: str) -> str:
        logger.debug(f'Server: objective_function: c_str: {c_str} f_str: {f_str} kwargs_str: {kwargs_str}')

        configuration = json.loads(c_str, cls=BenchmarkDecoder)
        fidelity = json.loads(f_str, cls=BenchmarkDecoder)
        kwargs = json.loads(kwargs_str, cls=BenchmarkDecoder)

        result = self.benchmark.objective_function_test(configuration=configuration, fidelity=fidelity, **kwargs)
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def get_meta_information(self):
        logger.debug('Server: get_meta_info called')
        return json.dumps(self.benchmark.get_meta_information(), indent=None, cls=BenchmarkEncoder)

    @Pyro4.oneway   # in case call returns much later than daemon.shutdown
    def shutdown(self):
        logger.debug('Server: Shutting down...')
        Pyro4.config.COMMTIMEOUT = 0.5
        self.pyro_running = False
        self.daemon.shutdown()


if __name__ == "__main__":
    Pyro4.config.REQUIRE_EXPOSE = False

    parser = argparse.ArgumentParser(prog='server_abstract_benchmark.py',
                                     description='HPOBench Container Server',
                                     usage='%(prog)s <importBase> <benchmark> <socket_id>')
    parser.add_argument('importBase', type=str,
                        help='Relative path to benchmark file in hpobench/benchmarks, e.g. ml.xgboost_benchmark')
    parser.add_argument('benchmark', type=str,
                        help='Classname of the benchmark, e.g. XGBoostBenchmark')
    parser.add_argument('socket_id', type=str,
                        help='socket_id for pyro-server')

    args = parser.parse_args()

    # pylint: disable=logging-fstring-interpolation
    exec(f"from hpobench.benchmarks.{args.importBase} import {args.benchmark} as Benchmark")
    bp = BenchmarkServer(args.socket_id)
