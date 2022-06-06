from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark, LRBenchmarkBB, LRBenchmarkMF, \
    LRMOBenchmark, LRMOBenchmarkBB, LRMOBenchmarkMF
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF, \
    NNMOBenchmark, NNMOBenchmarkBB, NNMOBenchmarkMF
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark, RandomForestBenchmarkBB, \
    RandomForestBenchmarkMF, RandomForestMOBenchmark, RandomForestMOBenchmarkBB, \
    RandomForestMOBenchmarkMF
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF, \
    SVMMOBenchmark, SVMMOBenchmarkBB, SVMMOBenchmarkMF
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark, TabularMOBenchmark
from hpobench.benchmarks.ml.yahpo_benchmark import YAHPOGymMORawBenchmark, YAHPOGymRawBenchmark

try:
    # `xgboost` is from https://xgboost.readthedocs.io/en/latest/install.html#conda
    # and not part of the scikit-learn bundle and not a strict requirement for running HPOBench
    # for other spaces and also for tabular benchmarks
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark, XGBoostBenchmarkBB, \
        XGBoostBenchmarkMF, XGBoostMOBenchmark, XGBoostMOBenchmarkBB, XGBoostMOBenchmarkMF
    __all__ = [
        'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
        'LRMOBenchmark', 'LRMOBenchmarkBB', 'LRMOBenchmarkMF',
        'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF',
        'NNMOBenchmark', 'NNMOBenchmarkBB', 'NNMOBenchmarkMF',
        'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
        'RandomForestMOBenchmark', 'RandomForestMOBenchmarkBB', 'RandomForestMOBenchmarkMF',
        'SVMBenchmark', 'SVMBenchmarkMF', 'SVMBenchmarkBB',
        'SVMMOBenchmark', 'SVMMOBenchmarkMF', 'SVMMOBenchmarkBB',
        'XGBoostBenchmarkBB', 'XGBoostBenchmarkMF', 'XGBoostBenchmark',
        'XGBoostMOBenchmarkBB', 'XGBoostMOBenchmarkMF', 'XGBoostMOBenchmark',
        'TabularBenchmark', 'TabularMOBenchmark',
        'YAHPOGymMORawBenchmark', 'YAHPOGymRawBenchmark',
    ]
except (ImportError, AttributeError) as e:
    __all__ = [
        'LRBenchmark', 'LRBenchmarkBB', 'LRBenchmarkMF',
        'LRMOBenchmark', 'LRMOBenchmarkBB', 'LRMOBenchmarkMF',
        'NNBenchmark', 'NNBenchmarkBB', 'NNBenchmarkMF',
        'NNMOBenchmark', 'NNMOBenchmarkBB', 'NNMOBenchmarkMF',
        'RandomForestBenchmark', 'RandomForestBenchmarkBB', 'RandomForestBenchmarkMF',
        'RandomForestMOBenchmark', 'RandomForestMOBenchmarkBB', 'RandomForestMOBenchmarkMF',
        'SVMBenchmark', 'SVMBenchmarkMF', 'SVMBenchmarkBB',
        'SVMMOBenchmark', 'SVMMOBenchmarkMF', 'SVMMOBenchmarkBB',
        'TabularBenchmark', 'TabularMOBenchmark',
        'YAHPOGymMORawBenchmark', 'YAHPOGymRawBenchmark',

    ]
