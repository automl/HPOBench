#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Cartpole Benchmark from hpobench/benchmarks/rl/cartpole.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class CartpoleReduced(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'CartpoleReduced')
        kwargs['container_name'] = kwargs.get('container_name', 'cartpole')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(CartpoleReduced, self).__init__(**kwargs)


class CartpoleFull(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'CartpoleFull')
        kwargs['container_name'] = kwargs.get('container_name', 'cartpole')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(CartpoleFull, self).__init__(**kwargs)
