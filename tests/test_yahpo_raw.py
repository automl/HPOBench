import os

os.environ['LD_LIBRARY_PATH'] = \
    '/opt/R/4.0.5/lib/R/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/jvm/java-11-openjdk-amd64/lib/server'

from hpobench.benchmarks.ml.yahpo_benchmark import YAHPOGymRawBenchmark, YAHPOGymMORawBenchmark


def test_mo_benchmark():
    from yahpo_gym.local_config import LocalConfiguration
    local_config = LocalConfiguration()
    local_config.init_config(data_path='/home/lmmista-wap072/Dokumente/Code/Data_HPOBench/Data/yahpo/yahpo_data')

    # b = YAHPOGymMORawBenchmark(scenario="rbv2_super", instance="3")
    # cfg = b.get_configuration_space().get_default_configuration()
    # b.objective_function(cfg)

    b = YAHPOGymMORawBenchmark(scenario="iaml_xgboost", instance="40981",)
    cfg = b.get_configuration_space().get_default_configuration()
    b.objective_function(cfg)


if __name__ == '__main__':
    test_mo_benchmark()