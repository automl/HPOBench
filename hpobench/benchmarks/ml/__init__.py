from hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark, HistGBBenchmarkBB, HistGBBenchmarkMF
from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark, LRBenchmarkBB, LRBenchmarkMF
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark, RandomForestBenchmarkBB, \
    RandomForestBenchmarkMF
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

try:
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark, XGBoostBenchmarkBB, XGBoostBenchmarkMF
except ImportError:
    pass


__all__ = ['HistGBBenchmark', 'HistGBBenchmarkBB', 'HistGBBenchmarkMF',
           'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
           'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF',
           'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
           'SVMBenchmark', 'SVMBenchmarkBB', 'SVMBenchmarkMF',
           'TabularBenchmark',
           'XGBoostBenchmark', 'XGBoostBenchmarkBB', 'XGBoostBenchmarkMF',
           ]
