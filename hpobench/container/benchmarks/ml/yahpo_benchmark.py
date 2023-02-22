#!/usr/bin/python3
# -*- coding: utf-8 -*-

from hpobench.container.client_abstract_benchmark import AbstractMOBenchmarkClient, \
    AbstractBenchmarkClient


class YAHPOGymMORawBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'YAHPOGymMORawBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'yahpo_raw')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(YAHPOGymMORawBenchmark, self).__init__(**kwargs)


class YAHPOGymRawBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'YAHPOGymRawBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'yahpo_raw')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(YAHPOGymRawBenchmark, self).__init__(**kwargs)
