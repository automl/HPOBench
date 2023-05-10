#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Tabular Benchmark from hpobench/benchmarks/nas/tabular_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class SliceLocalizationBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SliceLocalizationBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(SliceLocalizationBenchmark, self).__init__(**kwargs)


class ProteinStructureBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ProteinStructureBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ProteinStructureBenchmark, self).__init__(**kwargs)


class NavalPropulsionBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NavalPropulsionBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NavalPropulsionBenchmark, self).__init__(**kwargs)


class ParkinsonsTelemonitoringBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParkinsonsTelemonitoringBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ParkinsonsTelemonitoringBenchmark, self).__init__(**kwargs)


class SliceLocalizationBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SliceLocalizationBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(SliceLocalizationBenchmarkOriginal, self).__init__(**kwargs)


class ProteinStructureBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ProteinStructureBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ProteinStructureBenchmarkOriginal, self).__init__(**kwargs)


class NavalPropulsionBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NavalPropulsionBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NavalPropulsionBenchmarkOriginal, self).__init__(**kwargs)


class ParkinsonsTelemonitoringBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParkinsonsTelemonitoringBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ParkinsonsTelemonitoringBenchmarkOriginal, self).__init__(**kwargs)


__all__ = ["SliceLocalizationBenchmark", "SliceLocalizationBenchmarkOriginal",
           "ProteinStructureBenchmark", "ProteinStructureBenchmarkOriginal",
           "NavalPropulsionBenchmark", "NavalPropulsionBenchmarkOriginal",
           "ParkinsonsTelemonitoringBenchmark", "ParkinsonsTelemonitoringBenchmarkOriginal"]
