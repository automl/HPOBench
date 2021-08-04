#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the svm surrogates Benchmark from hpobench/benchmarks/surrogates/svm_benchmark.py
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class SurrogateSVMBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SurrogateSVMBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'surrogate_svm')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.2')
        super(SurrogateSVMBenchmark, self).__init__(**kwargs)
