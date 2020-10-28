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

import Pyro4
import numpy as np
from ConfigSpace.read_and_write import json as csjson

from hpolib.config import HPOlibConfig

# Read in the verbosity level from the environment variable HPOLIB_DEBUG
log_level_str = os.environ.get('HPOLIB_DEBUG', 'false')
LOG_LEVEL = logging.DEBUG if log_level_str == 'true' else logging.INFO

console = logging.StreamHandler()
console.setLevel(LOG_LEVEL)

logger = logging.getLogger('BenchmarkServer')
logger.setLevel(LOG_LEVEL)
logger.addHandler(console)


class BenchmarkEncoder(json.JSONEncoder):
    """ Simple encoder to serialize numpy arrays and other things which are not directly serializable """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int):
            return int(o)
        if isinstance(o, np.float):
            return float(o)
        if isinstance(o, enum.Enum):
            return str(o)
        return json.JSONEncoder.default(self, o)


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class BenchmarkServer:
    def __init__(self, socket_id):
        self.pyro_running = True
        config = HPOlibConfig()
        self.benchmark = None

        self.socket_id = socket_id
        socket_path = config.socket_dir / (self.socket_id + "_unix.sock")
        if socket_path.exists():
            os.remove(socket_path)
        self.daemon = Pyro4.Daemon(unixsocket=str(socket_path))

        _ = self.daemon.register(self, self.socket_id + ".unixsock")

        # start the event loop of the server to wait for calls
        self.daemon.requestLoop(loopCondition=lambda: self.pyro_running)

    def init_benchmark(self, kwargs_str):
        try:
            if kwargs_str != "{}":
                kwargs = json.loads(kwargs_str)
                if 'rng' in kwargs and isinstance(kwargs['rng'], list):
                    (rnd0, rnd1, rnd2, rnd3, rnd4) = kwargs['rng']
                    rnd1 = [np.uint32(number) for number in rnd1]
                    kwargs['rng'] = np.random.set_state((rnd0, rnd1, rnd2, rnd3, rnd4))
                    logger.debug('Server: Rng works')
                self.benchmark = Benchmark(**kwargs)  # noqa: F821
            else:
                self.benchmark = Benchmark()  # noqa: F821
            logger.info('Server: Connected Successfully')
        except Exception as exception:
            logger.exception(exception)

    def get_configuration_space(self, kwargs_str: str) -> str:
        logger.debug(f'Server: get_config_space: kwargs_str: {kwargs_str}')

        kwargs = json.loads(kwargs_str)
        seed = kwargs.get('seed', None)

        result = self.benchmark.get_configuration_space(seed=seed)
        logger.debug(f'Server: Configspace: {result}')
        return csjson.write(result, indent=None)

    def get_fidelity_space(self, kwargs_str: str) -> str:
        logger.debug(f'Server: get_fidelity_space: kwargs_str: {kwargs_str}')

        kwargs = json.loads(kwargs_str)
        seed = kwargs.get('seed', None)

        result = self.benchmark.get_fidelity_space(seed=seed)
        logger.debug(f'Server: Fidelity Space: {result}')
        return csjson.write(result, indent=None)

    def objective_function_list(self, c_str: str, f_str: str, kwargs_str: str) -> str:
        configuration = json.loads(c_str)
        fidelity = json.loads(f_str)
        kwargs = json.loads(kwargs_str)

        result = self.benchmark.objective_function(configuration=configuration, fidelity=fidelity, **kwargs)
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test_list(self, c_str: str, f_str: str, kwargs_str: str) -> str:
        configuration = json.loads(c_str)
        fidelity = json.loads(f_str)
        kwargs = json.loads(kwargs_str)

        result = self.benchmark.objective_function_test(configuration=configuration, fidelity=fidelity, **kwargs)
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function(self, c_str: str, f_str: str, kwargs_str: str) -> str:
        logger.debug(f'Server: objective_function: c_str: {c_str} f_str: {f_str} kwargs_str: {kwargs_str}')

        configuration = json.loads(c_str)
        fidelity = json.loads(f_str)
        kwargs = json.loads(kwargs_str)

        result = self.benchmark.objective_function(configuration=configuration, fidelity=fidelity, **kwargs)
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def objective_function_test(self, c_str: str, f_str: str, kwargs_str: str) -> str:
        logger.debug(f'Server: objective_function: c_str: {c_str} f_str: {f_str} kwargs_str: {kwargs_str}')

        configuration = json.loads(c_str)
        fidelity = json.loads(f_str)
        kwargs = json.loads(kwargs_str)

        result = self.benchmark.objective_function_test(configuration=configuration, fidelity=fidelity, **kwargs)
        return json.dumps(result, indent=None, cls=BenchmarkEncoder)

    def get_meta_information(self):
        logger.debug('Server: get_meta_info called')
        return json.dumps(self.benchmark.get_meta_information(), indent=None)

    @Pyro4.oneway   # in case call returns much later than daemon.shutdown
    def shutdown(self):
        logger.debug('Server: Shutting down...')
        Pyro4.config.COMMTIMEOUT = 0.5
        self.pyro_running = False
        self.daemon.shutdown()


if __name__ == "__main__":
    Pyro4.config.REQUIRE_EXPOSE = False

    parser = argparse.ArgumentParser(prog='server_abstract_benchmark.py',
                                     description='HPOlib2 Container Server',
                                     usage='%(prog)s <importBase> <benchmark> <socket_id>')
    parser.add_argument('importBase', type=str,
                        help='Relative path to benchmark file in hpolib/benchmarks, e.g. ml.xgboost_benchmark')
    parser.add_argument('benchmark', type=str,
                        help='Classname of the benchmark, e.g. XGBoostBenchmark')
    parser.add_argument('socket_id', type=str,
                        help='socket_id for pyro-server')

    args = parser.parse_args()

    # pylint: disable=logging-fstring-interpolation
    exec(f"from hpolib.benchmarks.{args.importBase} import {args.benchmark} as Benchmark")
    bp = BenchmarkServer(args.socket_id)
