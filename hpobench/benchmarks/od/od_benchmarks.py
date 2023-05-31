"""
This is just an entry point for starting the benchmarks.
"""


from hpobench.benchmarks.od.od_ae import ODAutoencoder
from hpobench.benchmarks.od.od_kde import ODKernelDensityEstimation
from hpobench.benchmarks.od.od_ocsvm import ODOneClassSupportVectorMachine

__all__ = [ODAutoencoder, ODKernelDensityEstimation, ODOneClassSupportVectorMachine]
