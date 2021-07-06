#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for OCSVM and outlier detection """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class ODKernelDensityEstimation(AbstractBenchmarkClient):
    def __init__(self, dataset_name: str, **kwargs):
        kwargs['dataset_name'] = dataset_name
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ODKernelDensityEstimation')
        kwargs['container_name'] = kwargs.get('container_name', 'kde_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(ODKernelDensityEstimation, self).__init__(**kwargs)
