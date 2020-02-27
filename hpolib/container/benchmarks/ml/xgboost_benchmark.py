#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGBoost Benchmark from hpolib/benchmarks/ml/xgboost_benchmark """

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        # benchmark_name must be the exact same as the suffix in therecipe name (Singuarity.XGBoostBenchmark)
        super(XGBoostBenchmark, self).__init__()
        self.benchmark_name = kwargs.get('container_name', 'XGBoostBenchmark')
        self._setup(**kwargs)
