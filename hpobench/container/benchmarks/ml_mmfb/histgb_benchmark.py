#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the HistGB Benchmarks from hpobench/benchmarks/ml_mmfb/histgb_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class HistGBSearchSpace0Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBSearchSpace0Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBSearchSpace0Benchmark, self).__init__(**kwargs)


class HistGBSearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBSearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBSearchSpace1Benchmark, self).__init__(**kwargs)


class HistGBSearchSpace2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBSearchSpace2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBSearchSpace2Benchmark, self).__init__(**kwargs)


class HistGBSearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBSearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBSearchSpace3Benchmark, self).__init__(**kwargs)


__all__ = [HistGBSearchSpace0Benchmark, HistGBSearchSpace1Benchmark,
           HistGBSearchSpace2Benchmark, HistGBSearchSpace3Benchmark]
