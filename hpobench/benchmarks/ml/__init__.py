from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark, LRBenchmarkBB, LRBenchmarkMF
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark, RandomForestBenchmarkBB, \
    RandomForestBenchmarkMF
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark


try:
    # `xgboost` is from https://xgboost.readthedocs.io/en/latest/install.html#conda
    # and not part of the scikit-learn bundle and not a strict requirement for running HPOBench
    # for other spaces and also for tabular benchmarks
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark, XGBoostBenchmarkBB, XGBoostBenchmarkMF
    __all__ = [
        'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
        'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF',
        'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
        'SVMBenchmark', 'SVMBenchmarkBB', 'SVMBenchmarkMF',
        'XGBoostBenchmark', 'XGBoostBenchmarkBB', 'XGBoostBenchmarkMF',
        'TabularBenchmark',
    ]
except (ImportError, AttributeError) as e:
    __all__ = [
        'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
        'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF',
        'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
        'SVMBenchmark', 'SVMBenchmarkBB', 'SVMBenchmarkMF',
        'TabularBenchmark',
    ]
