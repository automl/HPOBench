#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Neural Network Benchmarks from hpobench/benchmarks/ml_mmfb/nn_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class NNBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNBenchmark, self).__init__(**kwargs)


class NNBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNBenchmarkBB, self).__init__(**kwargs)


class NNBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNBenchmarkMF, self).__init__(**kwargs)


__all__ = ['NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF']
