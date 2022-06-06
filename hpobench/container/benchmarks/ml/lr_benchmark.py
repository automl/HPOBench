#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the learning rate Benchmarks from hpobench/benchmarks/ml_mmfb/lr_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


container_name = "ml_mmfb"
container_version = "0.0.4"


class LRBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(LRBenchmark, self).__init__(**kwargs)


class LRBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(LRBenchmarkBB, self).__init__(**kwargs)


class LRBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(LRBenchmarkMF, self).__init__(**kwargs)


class LRMOBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRMOBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(LRMOBenchmark, self).__init__(**kwargs)


class LRMOBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRMOBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(LRMOBenchmarkBB, self).__init__(**kwargs)


class LRMOBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRMOBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(LRMOBenchmarkMF, self).__init__(**kwargs)


__all__ = [
    'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
    'LRMOBenchmark', 'LRMOBenchmarkBB', 'LRMOBenchmarkMF'
]
