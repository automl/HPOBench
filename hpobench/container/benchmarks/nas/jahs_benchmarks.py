#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Benchmark for the NasBench201 Benchmark from hpobench/benchmarks/nas/jahs_benchmarks.py """

from hpobench.container.client_abstract_benchmark import AbstractMOBenchmarkClient, \
    AbstractBenchmarkClient as AbstractSOBenchmarkClient


# ######################### Single Objective - Surrogate ###############################################################
class JAHSSOCifar10SurrogateBenchmark(AbstractSOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSSOCifar10SurrogateBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSSOCifar10SurrogateBenchmark, self).__init__(**kwargs)


class JAHSSOColorectalHistologySurrogateBenchmark(AbstractSOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSSOColorectalHistologySurrogateBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSSOColorectalHistologySurrogateBenchmark, self).__init__(**kwargs)


class JAHSSOFashionMNISTSurrogateBenchmark(AbstractSOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSSOFashionMNISTSurrogateBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSSOFashionMNISTSurrogateBenchmark, self).__init__(**kwargs)
# ######################### Single Objective - Surrogate ###############################################################


# ######################### Single Objective - Tabular #################################################################
class JAHSSOCifar10TabularBenchmark(AbstractSOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSSOCifar10TabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSSOCifar10TabularBenchmark, self).__init__(**kwargs)


class JAHSSOColorectalHistologyTabularBenchmark(AbstractSOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSSOColorectalHistologyTabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSSOColorectalHistologyTabularBenchmark, self).__init__(**kwargs)


class JAHSSOFashionMNISTTabularBenchmark(AbstractSOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSSOFashionMNISTTabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSSOFashionMNISTTabularBenchmark, self).__init__(**kwargs)
# ######################### Single Objective - Tabular #################################################################


# ######################### Multi Objective - Surrogate ################################################################
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
# ######################### Multi Objective - Surrogate ################################################################


# ######################### Multi Objective - Tabular ##################################################################
class JAHSMOCifar10TabularBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSMOCifar10TabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSMOCifar10TabularBenchmark, self).__init__(**kwargs)


class JAHSMOColorectalHistologyTabularBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSMOColorectalHistologyTabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSMOColorectalHistologyTabularBenchmark, self).__init__(**kwargs)


class JAHSMOFashionMNISTTabularBenchmark(AbstractMOBenchmarkClient):
    def __init__(self, **kwargs):
        kwargs['benchmark_name'] = kwargs.get('benchmark_name', 'JAHSMOFashionMNISTTabularBenchmark')
        kwargs['container_name'] = kwargs.get('container_name', 'jahs_benchmark')
        kwargs['latest'] = kwargs.get('container_tag', '0.0.1')
        super(JAHSMOFashionMNISTTabularBenchmark, self).__init__(**kwargs)
# ######################### Multi Objective - Tabular ##################################################################


__all__ = [
    "JAHSSOCifar10SurrogateBenchmark",
    "JAHSSOColorectalHistologySurrogateBenchmark",
    "JAHSSOFashionMNISTSurrogateBenchmark",

    "JAHSSOCifar10TabularBenchmark",
    "JAHSSOColorectalHistologyTabularBenchmark",
    "JAHSSOFashionMNISTTabularBenchmark",

    "JAHSMOCifar10SurrogateBenchmark",
    "JAHSMOColorectalHistologySurrogateBenchmark",
    "JAHSMOFashionMNISTSurrogateBenchmark",

    "JAHSMOCifar10TabularBenchmark",
    "JAHSMOColorectalHistologyTabularBenchmark",
    "JAHSMOFashionMNISTTabularBenchmark",
]