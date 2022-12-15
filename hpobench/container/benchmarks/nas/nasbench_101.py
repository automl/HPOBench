#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Tabular Benchmark from hpobench/benchmarks/nas/nasbench_101.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient, AbstractMOBenchmarkClient


class NASCifar10ABenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10ABenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NASCifar10ABenchmark, self).__init__(**kwargs)


class NASCifar10BBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10BBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NASCifar10BBenchmark, self).__init__(**kwargs)


class NASCifar10CBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10CBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NASCifar10CBenchmark, self).__init__(**kwargs)


class NASCifar10AMOBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10AMOBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NASCifar10AMOBenchmark, self).__init__(**kwargs)


class NASCifar10BMOBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10BMOBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NASCifar10BMOBenchmark, self).__init__(**kwargs)


class NASCifar10CMOBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NASCifar10CMOBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_101')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NASCifar10CMOBenchmark, self).__init__(**kwargs)

