import sys
import pytest

from hpobench.container.benchmarks.ml.pybnn import BNNOnToyFunction, BNNOnBostonHousing, BNNOnProteinStructure, \
    BNNOnYearPrediction

from hpobench.util.container_utils import enable_container_debug
from hpobench.util.test_utils import check_run_all_tests, DEFAULT_SKIP_MSG

enable_container_debug()
MSG = 'Skip this test for new (>3.9) python versions. ' \
      'The paramnet benchmarks require an specific old scikit learn version. This version however does not work under ' \
      'python 3.10. Therefore we skip this test. The containerized version does still work under 3.10.'


@pytest.mark.skipif(sys.version_info > (3, 9), reason=MSG)
def test_bnn_init():
    benchmark = BNNOnToyFunction(rng=1)

    fs = benchmark.get_fidelity_space(seed=0)
    fidelity = fs.sample_configuration().get_dictionary()
    assert fidelity['budget'] == 5714

    meta = benchmark.get_meta_information()
    assert meta is not None

    cs = benchmark.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()

    assert config['l_rate'] == pytest.approx(0.0037, abs=0.001)
    assert config['burn_in'] == pytest.approx(0.43905, abs=0.001)
    assert config['n_units_1'] == 104
    assert config['n_units_2'] == 68
    assert config['mdecay'] == pytest.approx(0.6027, abs=0.001)

    result = benchmark.objective_function(configuration=config, fidelity=fidelity, rng=1)
    assert result['function_value'] == pytest.approx(380.08, abs=0.1)
    assert result['cost'] > 1
    assert result['info']['fidelity']['budget'] == 5714

    result = benchmark.objective_function_test(configuration=config)
    assert result['function_value'] == pytest.approx(183.6146, abs=0.1)
    assert result['cost'] is not None
    # test if budget is maximal:
    assert result['info']['fidelity']['budget'] == 10000


def simple_call(benchmark):
    cs = benchmark.get_configuration_space(seed=0)
    config = cs.sample_configuration().get_dictionary()

    fidelity = {'budget': 1000}

    result = benchmark.objective_function(configuration=config, fidelity=fidelity, rng=1)
    return result


def test_bnn_boston_housing():
    benchmark = BNNOnBostonHousing(rng=1)
    test_result = simple_call(benchmark)
    assert test_result['function_value'] == pytest.approx(1262.0869, abs=0.1)
    assert test_result['cost'] > 0
    assert test_result['info']['fidelity']['budget'] == 1000


@pytest.mark.skipif(not check_run_all_tests(), reason=DEFAULT_SKIP_MSG)
def test_bnn_protein():
    benchmark = BNNOnProteinStructure(rng=1)
    test_result = simple_call(benchmark)
    assert test_result['function_value'] == pytest.approx(1050.5733, abs=0.1)
    assert test_result['cost'] > 0
    assert test_result['info']['fidelity']['budget'] == 1000


@pytest.mark.skipif(not check_run_all_tests(), reason=DEFAULT_SKIP_MSG)
def test_year_pred():
    benchmark = BNNOnYearPrediction(rng=1)
    test_result = simple_call(benchmark)
    assert test_result['function_value'] == pytest.approx(2105.2726, abs=0.1)
    assert test_result['cost'] > 0
    assert test_result['info']['fidelity']['budget'] == 1000
