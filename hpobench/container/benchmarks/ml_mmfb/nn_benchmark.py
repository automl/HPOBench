#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Neural Network Benchmarks from hpobench/benchmarks/ml_mmfb/nn_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class NNSearchSpace0Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNSearchSpace0Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNSearchSpace0Benchmark, self).__init__(**kwargs)


class NNSearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNSearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNSearchSpace1Benchmark, self).__init__(**kwargs)


class NNSearchSpace2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNSearchSpace2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNSearchSpace2Benchmark, self).__init__(**kwargs)


class NNSearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NNSearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(NNSearchSpace3Benchmark, self).__init__(**kwargs)


__all__ = [NNSearchSpace0Benchmark, NNSearchSpace1Benchmark,
           NNSearchSpace2Benchmark, NNSearchSpace3Benchmark]
