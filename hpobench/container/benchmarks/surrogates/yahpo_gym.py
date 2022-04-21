#!/usr/bin/python3
# -*- coding: utf-8 -*-

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

class YAHPOGymBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'YAHPOGymBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'yahpo_gym')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(YAHPOGymBenchmark, self).__init__(**kwargs)
