#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGBoost Benchmark from hpolib/benchmarks/ml/xgboost_benchmark """

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        super(XGBoostBenchmark, self).__init__()
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'xgboost_benchmark')
        self._setup(**kwargs)
