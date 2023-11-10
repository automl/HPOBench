#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Cartpole Benchmark from hpobench/benchmarks/rl/cartpole.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from hpobench.util.container_utils import get_container_version


container_name = "cartpole"
container_version = get_container_version(container_name)


class CartpoleReduced(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'CartpoleReduced')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(CartpoleReduced, self).__init__(**kwargs)


class CartpoleFull(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'CartpoleFull')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(CartpoleFull, self).__init__(**kwargs)
