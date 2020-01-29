#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import importlib
from time import time

from hpolib.config import HPOlibConfig

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: server.py <importBase> <benchmark>")
        sys.exit()
    importBase = sys.argv[1]
    benchmark = sys.argv[2]

    module = importlib.import_module(f'hpolib.benchmarks.{importBase}')
    Benchmark = getattr(module, benchmark)
    # exec("from hpolib.benchmarks.%s import %s as Benchmark" % (importBase, benchmark))
    config = HPOlibConfig()
    start = time()
    b = Benchmark()
    print(f"Data download done, took totally {time()-start:.2f} s")
