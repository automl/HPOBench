""" Benchmark for the Multi-Objective CNN Benchmark from hpobench/benchmarks/mo/cnn_benchmark.py
"""

from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient


class FlowerCNNBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'FlowerCNNBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'mo_cnn')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(FlowerCNNBenchmark, self).__init__(**kwargs)


class FashioCNNBenchmark(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'FashionCNNBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'mo_cnn')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.4')
        super(FashioCNNBenchmark, self).__init__(**kwargs)
