#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the YAHPO surrogates Benchmark from hpobench/benchmarks/surrogates/yahpo_gym.py
Test with
from hpobench.container.benchmarks.surrogates.yahpo_gym import YAHPOGymBenchmark
b = YAHPOGymBenchmark(container_source=".", container_name="yahpo_gym", scenario = "lcbench", instance = "3945", objective ="val_accuracy")
res = b.objective_function(configuration=b.get_configuration_space(seed=1).sample_configuration())
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class YAHPOGymBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'YAHPOGymBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'yahpo_gym')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(YAHPOGymBenchmark, self).__init__(**kwargs)
