#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGB Benchmarks from hpobench/benchmarks/ml_mmfb/xgboost_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostSearchSpace0Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostSearchSpace0Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(XGBoostSearchSpace0Benchmark, self).__init__(**kwargs)


class XGBoostSearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostSearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(XGBoostSearchSpace1Benchmark, self).__init__(**kwargs)


class XGBoostSearchSpace2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostSearchSpace2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(XGBoostSearchSpace2Benchmark, self).__init__(**kwargs)


class XGBoostSearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostSearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(XGBoostSearchSpace3Benchmark, self).__init__(**kwargs)


__all__ = [XGBoostSearchSpace0Benchmark, XGBoostSearchSpace1Benchmark,
           XGBoostSearchSpace2Benchmark, XGBoostSearchSpace3Benchmark]
