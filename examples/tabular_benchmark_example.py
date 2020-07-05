import argparse
from copy import deepcopy
from pathlib import Path
from time import time

import json_tricks

from hpolib.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark as TabBenchmark
from hpolib.benchmarks.nas.nasbench import NASCifar10ABenchmark as NasBenchmark


def run_experiment(on_travis=False):
    """
    benchmark = NasBenchmark(data_path='/home/philipp/Dokumente/Code/TabularBenchmarks')

    cs = benchmark.get_configuration_space()
    config = cs.get_default_configuration()

    print(config)

    result_dict_1 = benchmark.objective_function(configuration=config.get_dictionary())
    result_dict_2 = benchmark.objective_function(configuration=config)
    print(result_dict_1, result_dict_2)
    """

    # Tabular Benchmark
    benchmark = TabBenchmark(data_path='/home/philipp/Dokumente/Code/TabularBenchmarks/fcnet_tabular_benchmarks')

    cs = benchmark.get_configuration_space()
    config = cs.get_default_configuration()

    print(config)

    result_dict_1 = benchmark.objective_function(configuration=config.get_dictionary())
    result_dict_2 = benchmark.objective_function(configuration=config)

    print(result_dict_1, result_dict_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Learna on RFAM')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)


