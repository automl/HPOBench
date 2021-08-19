from hpobench.container.benchmarks.ml.histgb_benchmark import HistGBBenchmark, HistGBBenchmarkBB, HistGBBenchmarkMF
from hpobench.container.benchmarks.ml.lr_benchmark import LRBenchmark, LRBenchmarkBB, LRBenchmarkMF
from hpobench.container.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF
from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmark, RandomForestBenchmarkBB, \
    RandomForestBenchmarkMF
from hpobench.container.benchmarks.ml.svm_benchmark import SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF
from hpobench.container.benchmarks.ml.tabular_benchmark import TabularBenchmark
from hpobench.container.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark, XGBoostBenchmarkBB, XGBoostBenchmarkMF


__all__ = ['HistGBBenchmark', 'HistGBBenchmarkBB', 'HistGBBenchmarkMF',
           'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
           'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF',
           'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
           'SVMBenchmark', 'SVMBenchmarkBB', 'SVMBenchmarkMF',
           'TabularBenchmark',
           'XGBoostBenchmark', 'XGBoostBenchmarkBB', 'XGBoostBenchmarkMF']
