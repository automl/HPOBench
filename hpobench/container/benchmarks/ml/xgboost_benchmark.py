#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGBoost Benchmark from hpobench/benchmarks/ml/xgboost_benchmark """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostBenchmark(AbstractBenchmarkClient):
    def __init__(self, task_id: int, **kwargs):
        kwargs['task_id'] = task_id
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'xgboost_benchmark')
        super(XGBoostBenchmark, self).__init__(**kwargs)


class XGBoostBoosterBenchmark(AbstractBenchmarkClient):
    def __init__(self, task_id: int, **kwargs):
        kwargs['task_id'] = task_id
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBoosterBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'xgboost_benchmark')
        super(XGBoostBoosterBenchmark, self).__init__(**kwargs)
