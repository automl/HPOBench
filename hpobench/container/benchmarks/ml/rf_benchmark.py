#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Random Forest Benchmarks from hpobench/benchmarks/ml_mmfb/rf_benchmark.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class RandomForestBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestBenchmark, self).__init__(**kwargs)



class RandomForestBenchmarkMO(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkMO')
        kwargs['container_name'] = kwargs.get('container_name', 'mo_ml_mmfb')
        
        #Need to be changed to the production registry
        kwargs['container_source'] = 'oras://gitlab.tf.uni-freiburg.de:5050/sharmaa/hpobench-registry'
        kwargs['container_tag'] = '0.0.6'
        kwargs['latest'] = '0.0.6' 
        # kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        # kwargs['latest'] = kwargs.get('container_tag', '0.0.2')
        super(RandomForestBenchmarkMO, self).__init__(**kwargs)


class RandomForestBenchmarkBB(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkBB')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestBenchmarkBB, self).__init__(**kwargs)


class RandomForestBenchmarkMF(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'RandomForestBenchmarkMF')
        kwargs['container_name'] = kwargs.get('container_name', 'ml_mmfb')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(RandomForestBenchmarkMF, self).__init__(**kwargs)


<<<<<<< HEAD
__all__ = ['RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF', 'RandomForestBenchmarkMO']
=======
__all__ = ['RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF', 'RandomForestBenchmarkMO']
>>>>>>> mo_tabular
