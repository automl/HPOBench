""" Benchmark for the Multi-Objective Language Model Benchmark from hpobench/benchmarks/mo/lm_benchmark.py
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class LanguageModelBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'LanguageModelBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'lm_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(LanguageModelBenchmark, self).__init__(**kwargs)
