#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the NasBench201 Benchmark from hpobench/benchmarks/nas/jahs_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractMOBenchmarkClient


class JAHSMOCifar10SurrogateBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSMOCifar10SurrogateBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSMOCifar10SurrogateBenchmark, self).__init__(**kwargs)


class JAHSMOColorectalHistologySurrogateBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSMOColorectalHistologySurrogateBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSMOColorectalHistologySurrogateBenchmark, self).__init__(**kwargs)


class JAHSMOFashionMNISTSurrogateBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSMOFashionMNISTSurrogateBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSMOFashionMNISTSurrogateBenchmark, self).__init__(**kwargs)


__all__ = [
    "JAHSMOCifar10SurrogateBenchmark",
    "JAHSMOColorectalHistologySurrogateBenchmark",
    "JAHSMOFashionMNISTSurrogateBenchmark",
]
