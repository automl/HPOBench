#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the SVM Benchmarks from hpobench/benchmarks/ml_mmfb/svm_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


container_name = "ml_mfbb"
container_version = "0.0.3"


class SVMBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(SVMBenchmark, self).__init__(**kwargs)


class SVMBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(SVMBenchmarkMF, self).__init__(**kwargs)


class SVMBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(SVMBenchmarkBB, self).__init__(**kwargs)


__all__ = ['SVMBenchmark', 'SVMBenchmarkMF', 'SVMBenchmarkBB']
