"""
Tabular benchmark
=================

This examples shows the usage of the containerized tabular benchmark.
To note: You don't have to pass the container name to the Benchmark-Constructor. It is automatically set, but for
demonstration purpose, we show how to set it.

container_source can be either a path to a registry (e.g. sylabs.io, singularity_hub.org) or a local path on your local
file system. If it is a link to a registry, the container will be downloaded to the default data dir, set in the
hpobenchrc. A second call, will first look into the data directory, if the container is already available, so it will not
be downloaded twice.

Please install the necessary dependencies via ``pip install .`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""

import argparse

from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark as TabBenchmarkContainer


def run_experiment(on_travis=False):

    benchmark = TabBenchmarkContainer(container_name='tabular_benchmarks',
                                      container_source='library://phmueller/automl',
                                      rng=1)

    cs = benchmark.get_configuration_space(seed=1)
    config = cs.sample_configuration()
    print(config)

    # You can pass the configuration either as a dictionary or a ConfigSpace.configuration
    result_dict_1 = benchmark.objective_function(configuration=config.get_dictionary())
    result_dict_2 = benchmark.objective_function(configuration=config)
    print(result_dict_1, result_dict_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TabularNad')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)
