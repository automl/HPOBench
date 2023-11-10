#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the HistGB Benchmarks from hpobench/benchmarks/ml_mmfb/histgb_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from hpobench.util.container_utils import get_container_version


container_name = "ml_mmfb"
container_version = get_container_version(container_name)



class HistGBBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(HistGBBenchmark, self).__init__(**kwargs)


class HistGBBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(HistGBBenchmarkBB, self).__init__(**kwargs)


class HistGBBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'HistGBBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(HistGBBenchmarkMF, self).__init__(**kwargs)


__all__ = ['HistGBBenchmark', 'HistGBBenchmarkBB', 'HistGBBenchmarkMF']
