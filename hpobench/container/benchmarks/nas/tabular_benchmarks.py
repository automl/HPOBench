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


class SliceLocalizationOriginalBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'SliceLocalizationOriginalBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(SliceLocalizationOriginalBenchmark, self).__init__(**kwargs)


class ProteinStructureOriginalBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ProteinStructureOriginalBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ProteinStructureOriginalBenchmark, self).__init__(**kwargs)


class NavalPropulsionOriginalBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'NavalPropulsionOriginalBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(NavalPropulsionOriginalBenchmark, self).__init__(**kwargs)


class ParkinsonsTelemonitoringOriginalBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['data_path'] = '/home/fcnet_tabular_benchmarks'
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ParkinsonsTelemonitoringOriginalBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'tabular_benchmarks')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ParkinsonsTelemonitoringOriginalBenchmark, self).__init__(**kwargs)


__all__ = ["SliceLocalizationBenchmark", "SliceLocalizationOriginalBenchmark",
           "ProteinStructureBenchmark", "ProteinStructureOriginalBenchmark",
           "NavalPropulsionBenchmark", "NavalPropulsionOriginalBenchmark",
           "ParkinsonsTelemonitoringBenchmark", "ParkinsonsTelemonitoringOriginalBenchmark"]
