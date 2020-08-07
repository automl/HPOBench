#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Tabular Benchmark from hpolib/benchmarks/nas/nasbench_101.py """

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class NASCifar10ABenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10ABenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        super(NASCifar10ABenchmark, self).__init__(**kwargs)


class NASCifar10BBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10BBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        super(NASCifar10BBenchmark, self).__init__(**kwargs)


class NASCifar10CBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/data'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10CBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        super(NASCifar10CBenchmark, self).__init__(**kwargs)
