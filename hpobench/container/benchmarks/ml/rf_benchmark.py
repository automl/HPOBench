#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Random Forest Benchmarks from hpobench/benchmarks/ml_mmfb/rf_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


container_name = "ml_mmfb"
container_version = "0.0.4"


class RandomForestBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(RandomForestBenchmark, self).__init__(**kwargs)


class RandomForestBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(RandomForestBenchmarkBB, self).__init__(**kwargs)


class RandomForestBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(RandomForestBenchmarkMF, self).__init__(**kwargs)


class RandomForestMOBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestMOBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(RandomForestMOBenchmark, self).__init__(**kwargs)


class RandomForestMOBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestMOBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(RandomForestMOBenchmarkBB, self).__init__(**kwargs)


class RandomForestMOBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestMOBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(RandomForestMOBenchmarkMF, self).__init__(**kwargs)


__all__ = [
    'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
    'RandomForestMOBenchmark', 'RandomForestMOBenchmarkBB', 'RandomForestMOBenchmarkMF'
]
