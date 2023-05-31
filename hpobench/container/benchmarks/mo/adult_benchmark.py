""" Benchmark for the Multi-Objective Adult Benchmark from hpobench/benchmarks/mo/adult_benchmark.py
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class AdultBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'AdultBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'fair_adult')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.2')
        super(AdultBenchmark, self).__init__(**kwargs)
