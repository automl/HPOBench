#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the HistGB Benchmarks from hpobench/benchmarks/ml_mmfb/histgb_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class HistGBBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBBenchmark, self).__init__(**kwargs)


class HistGBBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBBenchmarkBB, self).__init__(**kwargs)


class HistGBBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(HistGBBenchmarkMF, self).__init__(**kwargs)


__all__ = ['HistGBBenchmark', 'HistGBBenchmarkBB', 'HistGBBenchmarkMF']
