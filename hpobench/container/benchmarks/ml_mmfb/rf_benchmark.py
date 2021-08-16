#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Random Forest Benchmarks from hpobench/benchmarks/ml_mmfb/rf_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class RandomForestSearchSpace0Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestSearchSpace0Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestSearchSpace0Benchmark, self).__init__(**kwargs)


class RandomForestSearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestSearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestSearchSpace1Benchmark, self).__init__(**kwargs)


class RandomForestSearchSpace2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestSearchSpace2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestSearchSpace2Benchmark, self).__init__(**kwargs)


class RandomForestSearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestSearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestSearchSpace3Benchmark, self).__init__(**kwargs)


__all__ = [RandomForestSearchSpace0Benchmark, RandomForestSearchSpace1Benchmark,
           RandomForestSearchSpace2Benchmark, RandomForestSearchSpace3Benchmark]
