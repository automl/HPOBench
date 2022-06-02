#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Tabular Benchmarks from hpobench/benchmarks/ml_mmfb/tabular_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


container_name = "ml_tabular_benchmarks"
container_version = "0.0.3"


class TabularBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'TabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(TabularBenchmark, self).__init__(**kwargs)


__all__ = ['TabularBenchmark']
