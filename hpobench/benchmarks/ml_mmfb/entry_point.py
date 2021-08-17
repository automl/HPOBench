from hpobench.benchmarks.ml_mmfb.histgb_benchmark import HistGBBenchmark, HistGBBenchmarkBB, HistGBBenchmarkMF
from hpobench.benchmarks.ml_mmfb.lr_benchmark import LRBenchmark, LRBenchmarkBB, LRBenchmarkMF
from hpobench.benchmarks.ml_mmfb.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF
from hpobench.benchmarks.ml_mmfb.rf_benchmark import RandomForestBenchmark, RandomForestBenchmarkBB, \
    RandomForestBenchmarkMF
from hpobench.benchmarks.ml_mmfb.svm_benchmark import SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF
from hpobench.benchmarks.ml_mmfb.tabular_benchmark import TabularBenchmark, OriginalTabularBenchmark
from hpobench.benchmarks.ml_mmfb.xgboost_benchmark import XGBoostBenchmark, XGBoostBenchmarkBB, XGBoostBenchmarkMF


__all__ = [HistGBBenchmark, HistGBBenchmarkBB, HistGBBenchmarkMF,
           LRBenchmark, LRBenchmarkBB, LRBenchmarkMF,
           NNBenchmark, NNBenchmarkBB, NNBenchmarkMF,
           RandomForestBenchmark, RandomForestBenchmarkBB, RandomForestBenchmarkMF,
           SVMBenchmark, SVMBenchmarkBB, SVMBenchmarkMF,
           TabularBenchmark, OriginalTabularBenchmark,
           XGBoostBenchmark, XGBoostBenchmarkBB, XGBoostBenchmarkMF]
