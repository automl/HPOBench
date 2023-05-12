#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the learning rate Benchmarks from hpobench/benchmarks/ml_mmfb/lr_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class LRBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRBenchmark, self).__init__(**kwargs)

class LRBenchmarkMO(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmarkMO')
        kwargs['container_name'] = kwargs.get('container_name', 'mo_ml_mmfb')
        kwargs['container_source'] = 'oras://gitlab.tf.uni-freiburg.de:5050/sharmaa/hpobench-registry'
        kwargs['container_tag'] = '0.0.6'
        kwargs['latest'] = '0.0.6' 
        # kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        # kwargs['latest'] = kwargs.get('container_tag', '0.0.2')
        super(LRBenchmarkMO, self).__init__(**kwargs)


class LRBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRBenchmarkBB, self).__init__(**kwargs)


class LRBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LRBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LRBenchmarkMF, self).__init__(**kwargs)


<<<<<<< HEAD
__all__ = ['LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF', 'LRBenchmarkMO']
=======
__all__ = ['LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF', 'LRBenchmarkMO']
>>>>>>> mo_tabular
