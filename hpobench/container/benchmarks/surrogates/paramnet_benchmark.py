#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Paramnet surrogates Benchmark from hpobench/benchmarks/surrogates/paramnet_benchmark.py
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class ParamNetAdultOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetAdultOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetAdultOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetAdultOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetAdultOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetAdultOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetHiggsOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetHiggsOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetHiggsOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetHiggsOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetHiggsOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetHiggsOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetLetterOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetLetterOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetLetterOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetLetterOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetLetterOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetLetterOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetMnistOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetMnistOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetMnistOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetMnistOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetMnistOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetMnistOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetOptdigitsOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetOptdigitsOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetOptdigitsOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetOptdigitsOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetOptdigitsOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetOptdigitsOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetPokerOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetPokerOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetPokerOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetPokerOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetPokerOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        super(ParamNetPokerOnTimeBenchmark, self).__init__(**kwargs)
