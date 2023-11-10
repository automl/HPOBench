#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the pybnn Benchmark from hpobench/benchmarks/ml/pybnn.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from hpobench.util.container_utils import get_container_version


container_name = "pybnn"
container_version = get_container_version(container_name)


class BNNOnToyFunction(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnToyFunction')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(BNNOnToyFunction, self).__init__(**kwargs)


class BNNOnBostonHousing(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnBostonHousing')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(BNNOnBostonHousing, self).__init__(**kwargs)


class BNNOnProteinStructure(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnProteinStructure')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(BNNOnProteinStructure, self).__init__(**kwargs)


class BNNOnYearPrediction(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'BNNOnYearPrediction')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(BNNOnYearPrediction, self).__init__(**kwargs)
