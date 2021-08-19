#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Random Forest Benchmarks from hpobench/benchmarks/ml_mmfb/rf_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class RandomForestBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestBenchmark, self).__init__(**kwargs)


class RandomForestBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestBenchmarkBB, self).__init__(**kwargs)


class RandomForestBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestBenchmarkMF, self).__init__(**kwargs)


__all__ = ['RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF']
