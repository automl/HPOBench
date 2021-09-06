#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGBoost Benchmark from hpobench/benchmarks/ml/xgboost_benchmark """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostBenchmark(AbstractBenchmarkClient):
    def __init__(self, task_id: int, **kwargs):
        kwargs['task_id'] = task_id
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'xgboost_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.3')
        super(XGBoostBenchmark, self).__init__(**kwargs)


class XGBoostExtendedBenchmark(AbstractBenchmarkClient):
    def __init__(self, task_id: int, **kwargs):
        kwargs['task_id'] = task_id
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'XGBoostExtendedBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'xgboost_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.3')
        super(XGBoostExtendedBenchmark, self).__init__(**kwargs)
