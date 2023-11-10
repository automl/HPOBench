#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for OCSVM and outlier detection """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from hpobench.util.container_utils import get_container_version


container_name = "outlier_detection"
container_version = get_container_version(container_name)


class ODAutoencoder(AbstractBenchmarkClient):
    def __init__(self, dataset_name: str, **kwargs):
        kwargs['dataset_name'] = dataset_name
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ODAutoencoder')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ODAutoencoder, self).__init__(**kwargs)


class ODKernelDensityEstimation(AbstractBenchmarkClient):
    def __init__(self, dataset_name: str, **kwargs):
        kwargs['dataset_name'] = dataset_name
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ODKernelDensityEstimation')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ODKernelDensityEstimation, self).__init__(**kwargs)


class ODOneClassSupportVectorMachine(AbstractBenchmarkClient):
    def __init__(self, dataset_name: str, **kwargs):
        kwargs['dataset_name'] = dataset_name
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ODOneClassSupportVectorMachine')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ODOneClassSupportVectorMachine, self).__init__(**kwargs)
