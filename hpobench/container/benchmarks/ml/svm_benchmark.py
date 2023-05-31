#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the SVM Benchmarks from hpobench/benchmarks/ml_mmfb/svm_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class SVMBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(SVMBenchmark, self).__init__(**kwargs)

class SVMBenchmarkMO(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmarkMO')
        kwargs['container_name'] = kwargs.get('container_name', 'mo_ml_mmfb')
        
        #Need to be changed to the production registry
        kwargs['container_source'] = 'oras://gitlab.tf.uni-freiburg.de:5050/sharmaa/hpobench-registry'
        kwargs['container_tag'] = '0.0.6'
        kwargs['latest'] = '0.0.6' 
        #kwargs['latest'] = kwargs.get('container_tag', '0.0.2')
        super(SVMBenchmarkMO, self).__init__(**kwargs)


class SVMBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(SVMBenchmarkMF, self).__init__(**kwargs)


class SVMBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SVMBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(SVMBenchmarkBB, self).__init__(**kwargs)


<<<<<<< HEAD
__all__ = ['SVMBenchmark', 'SVMBenchmarkMF', 'SVMBenchmarkBB', 'SVMBenchmarkMO']
=======
__all__ = ['SVMBenchmark', 'SVMBenchmarkMF', 'SVMBenchmarkBB', 'SVMBenchmarkMO']
>>>>>>> mo_tabular
