from hpobench.container.benchmarks.ml.yahpo_benchmark import YAHPOGymMORawBenchmark


def test_mo_benchmark():

    b = YAHPOGymMORawBenchmark(scenario="iaml_xgboost", instance="40981",)
    cfg = b.get_configuration_space().get_default_configuration()
    b.objective_function(cfg)


if __name__ == '__main__':
    test_mo_benchmark()