#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the rbv2 surrogates Benchmark from hpobench/benchmarks/surrogates/yahpo_gym.py
Test with

from hpobench.config import HPOBenchConfig
cdir = HPOBenchConfig().container_dir
from hpobench.container.benchmarks.ml.rbv2 import rbv2Benchmark
b = rbv2Benchmark(container_source = str(cdir), scenario = "rbv2_rpart", instance = "1040", objective = "logloss")
res = b.objective_function(configuration=b.get_configuration_space(seed=1).sample_configuration())
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class rbv2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'rbv2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'rbv2')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(rbv2Benchmark, self).__init__(**kwargs)
