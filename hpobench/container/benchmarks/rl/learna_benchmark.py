#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the learna benchmark from hpobench/benchmarks/rl/learna_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from hpobench.util.container_utils import get_container_version


container_name = "learna_benchmark"
container_version = get_container_version(container_name)


class Learna(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/learna/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'Learna')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(Learna, self).__init__(**kwargs)


class MetaLearna(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/learna/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'MetaLearna')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(MetaLearna, self).__init__(**kwargs)
