#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Paramnet surrogates Benchmark from hpobench/benchmarks/surrogates/paramnet_benchmark.py
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class ParamNetAdultOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetAdultOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetAdultOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetAdultOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetAdultOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetAdultOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetReducedAdultOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedAdultOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedAdultOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetReducedAdultOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedAdultOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedAdultOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetHiggsOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetHiggsOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetHiggsOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetHiggsOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetHiggsOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetHiggsOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetReducedHiggsOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedHiggsOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedHiggsOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetReducedHiggsOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedHiggsOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedHiggsOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetLetterOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetLetterOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetLetterOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetLetterOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetLetterOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetLetterOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetReducedLetterOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedLetterOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedLetterOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetReducedLetterOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedLetterOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedLetterOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetMnistOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetMnistOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetMnistOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetMnistOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetMnistOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetMnistOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetReducedMnistOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedMnistOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedMnistOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetReducedMnistOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedMnistOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedMnistOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetOptdigitsOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetOptdigitsOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetOptdigitsOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetOptdigitsOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetOptdigitsOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetOptdigitsOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetReducedOptdigitsOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedOptdigitsOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedOptdigitsOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetReducedOptdigitsOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedOptdigitsOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedOptdigitsOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetPokerOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetPokerOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetPokerOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetPokerOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetPokerOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetPokerOnTimeBenchmark, self).__init__(**kwargs)


class ParamNetReducedPokerOnStepsBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedPokerOnStepsBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedPokerOnStepsBenchmark, self).__init__(**kwargs)


class ParamNetReducedPokerOnTimeBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParamNetReducedPokerOnTimeBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'paramnet')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(ParamNetReducedPokerOnTimeBenchmark, self).__init__(**kwargs)
