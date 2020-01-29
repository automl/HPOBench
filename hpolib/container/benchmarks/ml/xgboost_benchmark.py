#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the XGBoost Benchmark from hpolib/benchmarks/ml/xgboost_benchmark """

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostOnMnist(AbstractBenchmarkClient):
    def __init__(self, **kwargs):

        # benchmark_name must be the exact same as the suffix in the recipe name (Singuarity.XGBoostOnMnist)
        super(XGBoostOnMnist, self).__init__()
        self.benchmark_name = 'XGBoostOnMnist'
        kwargs['img_source'] = kwargs.get('img_source', 'shub://PhMueller/Test')

        self._setup(**kwargs)

