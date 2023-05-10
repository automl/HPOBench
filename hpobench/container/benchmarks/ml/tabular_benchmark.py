#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Tabular Benchmarks from hpobench/benchmarks/ml_mmfb/tabular_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class TabularBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'TabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(TabularBenchmark, self).__init__(**kwargs)

class TabularBenchmarkMO(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'TabularBenchmarkMO')
        kwargs['container_name'] = kwargs.get('container_name', 'mo_ml_mmfb')
        kwargs['container_source'] = 'oras://gitlab.tf.uni-freiburg.de:5050/sharmaa/hpobench-registry'
        kwargs['container_tag'] = '0.0.8'
        kwargs['latest'] = '0.0.8'
        super(TabularBenchmarkMO, self).__init__(**kwargs)

__all__ = ['TabularBenchmark', 'TabularBenchmarkMO']
