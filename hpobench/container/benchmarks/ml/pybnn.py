#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the pybnn Benchmark from hpobench/benchmarks/ml/pybnn.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class BNNOnToyFunction(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnToyFunction')
        kwargs['container_name'] = kwargs.get('container_name', 'pybnn')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(BNNOnToyFunction, self).__init__(**kwargs)


class BNNOnBostonHousing(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnBostonHousing')
        kwargs['container_name'] = kwargs.get('container_name', 'pybnn')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(BNNOnBostonHousing, self).__init__(**kwargs)


class BNNOnProteinStructure(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnProteinStructure')
        kwargs['container_name'] = kwargs.get('container_name', 'pybnn')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(BNNOnProteinStructure, self).__init__(**kwargs)


class BNNOnYearPrediction(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnYearPrediction')
        kwargs['container_name'] = kwargs.get('container_name', 'pybnn')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(BNNOnYearPrediction, self).__init__(**kwargs)
