#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Neural Network Benchmarks from hpobench/benchmarks/ml_mmfb/nn_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


container_name = "ml_mmfb"
container_version = "0.0.4"


class NNBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(NNBenchmark, self).__init__(**kwargs)


class NNBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(NNBenchmarkBB, self).__init__(**kwargs)


class NNBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(NNBenchmarkMF, self).__init__(**kwargs)


__all__ = ['NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF']
