#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for OCSVM and outlier detection """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class ODOneClassSupportVectorMachine(AbstractBenchmarkClient):
    def __init__(self, dataset_name: str, **kwargs):
        kwargs['dataset_name'] = dataset_name
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ODOneClassSupportVectorMachine')
        kwargs['container_name'] = kwargs.get('container_name', 'ocsvm_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(ODOneClassSupportVectorMachine, self).__init__(**kwargs)
