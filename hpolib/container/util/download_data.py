#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import importlib
from time import time

from hpolib.config import HPOlibConfig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='download_data.py',
                                     description='HPOlib3 - Download Data',
                                     usage='%(prog)s <importBase> <benchmark> <task_id>')
    parser.add_argument('importBase', type=str,
                        help='Relative path to benchmark file in hpolib/benchmarks, e.g. ml.xgboost_benchmark')
    parser.add_argument('benchmark', type=str,
                        help='Classname of the benchmark, e.g. XGBoostOnMnist')
    parser.add_argument('task_id', type=int,
                        help='Optional - Task id for an openml task')
    args = parser.parse_args()

    module = importlib.import_module(f'hpolib.benchmarks.{args.importBase}')
    Benchmark = getattr(module, args.benchmark)

    config = HPOlibConfig()
    task_id = args.get('task_id', None)
    start = time()
    if task_id is not None:
        b = Benchmark(task_id)
    else:
        b = Benchmark()
    print(f"Data download done, took totally {time() - start:.2f} s")
