#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the Tabular Benchmark from hpobench/benchmarks/nas/tabular_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient
from hpobench.util.container_utils import get_container_version


container_name = "tabular_benchmarks"
container_version = get_container_version(container_name)


class SliceLocalizationBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SliceLocalizationBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(SliceLocalizationBenchmark, self).__init__(**kwargs)


class ProteinStructureBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ProteinStructureBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ProteinStructureBenchmark, self).__init__(**kwargs)


class NavalPropulsionBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NavalPropulsionBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(NavalPropulsionBenchmark, self).__init__(**kwargs)


class ParkinsonsTelemonitoringBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParkinsonsTelemonitoringBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ParkinsonsTelemonitoringBenchmark, self).__init__(**kwargs)


class SliceLocalizationBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SliceLocalizationBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(SliceLocalizationBenchmarkOriginal, self).__init__(**kwargs)


class ProteinStructureBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ProteinStructureBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ProteinStructureBenchmarkOriginal, self).__init__(**kwargs)


class NavalPropulsionBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NavalPropulsionBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(NavalPropulsionBenchmarkOriginal, self).__init__(**kwargs)


class ParkinsonsTelemonitoringBenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParkinsonsTelemonitoringBenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', container_name)
        kwargs['latest'] = kwargs.get('container_tag', container_version)
        super(ParkinsonsTelemonitoringBenchmarkOriginal, self).__init__(**kwargs)


__all__ = ["SliceLocalizationBenchmark", "SliceLocalizationBenchmarkOriginal",
           "ProteinStructureBenchmark", "ProteinStructureBenchmarkOriginal",
           "NavalPropulsionBenchmark", "NavalPropulsionBenchmarkOriginal",
           "ParkinsonsTelemonitoringBenchmark", "ParkinsonsTelemonitoringBenchmarkOriginal"]
