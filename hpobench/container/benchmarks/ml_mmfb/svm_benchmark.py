#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the SVM Benchmarks from hpobench/benchmarks/ml_mmfb/svm_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class SVMSearchSpace0Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMSearchSpace0Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(SVMSearchSpace0Benchmark, self).__init__(**kwargs)


class SVMSearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMSearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(SVMSearchSpace1Benchmark, self).__init__(**kwargs)


__all__ = [SVMSearchSpace0Benchmark, SVMSearchSpace1Benchmark]
