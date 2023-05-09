from hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark, HistGBBenchmarkBB, HistGBBenchmarkMF
from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark, LRBenchmarkBB, LRBenchmarkMF, LRBenchmarkMO
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF, NNBenchmarkMO
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark, RandomForestBenchmarkBB, \
    RandomForestBenchmarkMF, RandomForestBenchmarkMO
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF, SVMBenchmarkMO
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark, TabularBenchmarkMO

try:
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark, XGBoostBenchmarkBB, XGBoostBenchmarkMF, XGBoostBenchmarkMO
except ImportError:
    pass


__all__ = ['HistGBBenchmark', 'HistGBBenchmarkBB', 'HistGBBenchmarkMF',
           'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF', 'LRBenchmarkMO', 
           'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF', 'NNBenchmarkMO',
           'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF', 'RandomForestBenchmarkMO', 
           'SVMBenchmark', 'SVMBenchmarkBB', 'SVMBenchmarkMF', 'SVMBenchmarkMO',
           'TabularBenchmark', 'TabularBenchmarkMO',
           'XGBoostBenchmark', 'XGBoostBenchmarkBB', 'XGBoostBenchmarkMF', 'XGBoostBenchmarkMO',
           ]