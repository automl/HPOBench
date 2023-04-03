#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGB Benchmarks from hpobench/benchmarks/ml_mmfb/xgboost_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


container_name = "ml_mmfb"
container_version = "0.0.4"


class XGBoostBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(XGBoostBenchmark, self).__init__(**kwargs)


class XGBoostBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(XGBoostBenchmarkBB, self).__init__(**kwargs)


class XGBoostBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(XGBoostBenchmarkMF, self).__init__(**kwargs)


class XGBoostSearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostSearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(XGBoostSearchSpace3Benchmark, self).__init__(**kwargs)


__all__ = [
    'XGBoostBenchmark', 'XGBoostBenchmarkBB', 'XGBoostBenchmarkMF', 'XGBoostSearchSpace3Benchmark'
]
