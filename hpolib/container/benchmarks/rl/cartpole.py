#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Cartpole Benchmark from hpolib/benchmarks/rl/cartpole_hyperband.py """

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class CartpoleReduced(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        super(CartpoleReduced, self).__init__()
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'CartpoleReduced')
        kwargs['container_name'] = kwargs.get('container_name', 'cartpole')
        self._setup(**kwargs)


class CartpoleFull(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        super(CartpoleFull, self).__init__()
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'CartpoleFull')
        kwargs['container_name'] = kwargs.get('container_name', 'cartpole')
        self._setup(**kwargs)
