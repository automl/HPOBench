#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the nasbench 1shot1 benchmarks from hpobench/benchmarks/nas/nasbench_1shot1.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class NASBench1shot1SearchSpace1Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASBench1shot1SearchSpace1Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_1shot1')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(NASBench1shot1SearchSpace1Benchmark, self).__init__(**kwargs)


class NASBench1shot1SearchSpace2Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASBench1shot1SearchSpace2Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_1shot1')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(NASBench1shot1SearchSpace2Benchmark, self).__init__(**kwargs)


class NASBench1shot1SearchSpace3Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASBench1shot1SearchSpace3Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_1shot1')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(NASBench1shot1SearchSpace3Benchmark, self).__init__(**kwargs)
