#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the learning rate Benchmarks from hpobench/benchmarks/ml_mmfb/lr_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class LRSearchSpace0Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRSearchSpace0Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRSearchSpace0Benchmark, self).__init__(**kwargs)


class LRSearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRSearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRSearchSpace1Benchmark, self).__init__(**kwargs)


class LRSearchSpace2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRSearchSpace2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRSearchSpace2Benchmark, self).__init__(**kwargs)


class LRSearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRSearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRSearchSpace3Benchmark, self).__init__(**kwargs)


__all__ = [LRSearchSpace0Benchmark, LRSearchSpace1Benchmark,
           LRSearchSpace2Benchmark, LRSearchSpace3Benchmark]
