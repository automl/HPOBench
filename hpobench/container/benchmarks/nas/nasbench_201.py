#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the NasBench201 Benchmark from hpobench/benchmarks/nas/nasbench_201.py """

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class Cifar10ValidNasBench201Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'Cifar10ValidNasBench201Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_201')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(Cifar10ValidNasBench201Benchmark, self).__init__(**kwargs)


class Cifar100NasBench201Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'Cifar100NasBench201Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_201')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(Cifar100NasBench201Benchmark, self).__init__(**kwargs)


class ImageNetNasBench201Benchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ImageNetNasBench201Benchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_201')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ImageNetNasBench201Benchmark, self).__init__(**kwargs)


class Cifar10ValidNasBench201BenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'Cifar10ValidNasBench201BenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_201')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(Cifar10ValidNasBench201BenchmarkOriginal, self).__init__(**kwargs)


class Cifar100NasBench201BenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'Cifar100NasBench201BenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_201')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(Cifar100NasBench201BenchmarkOriginal, self).__init__(**kwargs)


class ImageNetNasBench201BenchmarkOriginal(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'ImageNetNasBench201BenchmarkOriginal')
        kwargs['container_name'] = kwargs.get('container_name', 'nasbench_201')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.5')
        super(ImageNetNasBench201BenchmarkOriginal, self).__init__(**kwargs)


__all__ = ["Cifar10ValidNasBench201Benchmark",
           "Cifar100NasBench201Benchmark",
           "ImageNetNasBench201Benchmark",
           "Cifar10ValidNasBench201BenchmarkOriginal",
           "Cifar100NasBench201BenchmarkOriginal",
           "ImageNetNasBench201BenchmarkOriginal"]
