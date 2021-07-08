"""
ADVANCED: Create multiple connections to a Benchmark
====================================================

Sometimes it is useful to have only a single container instance running for multiple experiments.
For example if starting a container is very expensive.

Since version 0.0.8, we can now start a benchmark container in a background process and connect to the benchmark from
various other processes.
This example shows how to create a benchmark and create a separate proxy connection to it.

The benchmark is based on Pyro4 and should handle a certain amount of simultaneous calls.

Also make sure not have to many open Pyro4.Proxy connections.
(See https://pyro4.readthedocs.io/en/stable/tipstricks.html#after-x-simultaneous-proxy-connections-pyro-seems-to-freeze-fix-release-your-proxies-when-you-can)

Please install the necessary dependencies via ``pip install .`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""

import argparse

from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark as TabBenchmarkContainer


def run_experiment(on_travis=False):

    # First, we start the benchmark. This generates the unix-socket (address) where the benchmark is reachable.
    benchmark = TabBenchmarkContainer(container_name='tabular_benchmarks',
                                      container_source='library://phmueller/automl',
                                      rng=1)

    print(benchmark.socket_id)

    # Now, we could use this `socket_id` to connect to the benchmark from another process. For simplicity, we just
    # create a new proxy connection to it.
    # Note that you don't have to specify the container name or other things, since we only connect to it.

    proxy_to_benchmark = TabBenchmarkContainer(socket_id=benchmark.socket_id)

    cs = proxy_to_benchmark.get_configuration_space(seed=1)
    config = cs.sample_configuration()
    print(config)

    # You can pass the configuration either as a dictionary or a ConfigSpace.configuration
    result_dict_1 = proxy_to_benchmark.objective_function(configuration=config.get_dictionary())
    result_dict_2 = proxy_to_benchmark.objective_function(configuration=config)
    print(result_dict_1, result_dict_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TabularNad')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)
