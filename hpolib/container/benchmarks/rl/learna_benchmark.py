#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the learna benchmark from hpolib/benchmarks/rl/learna_benchmarks.py """

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class Learna(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/learna/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'Learna')
        kwargs['container_name'] = kwargs.get('container_name', 'learna_benchmark')
        super(Learna, self).__init__(**kwargs)


class MetaLearna(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/learna/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'MetaLearna')
        kwargs['container_name'] = kwargs.get('container_name', 'learna_benchmark')
        super(MetaLearna, self).__init__(**kwargs)
