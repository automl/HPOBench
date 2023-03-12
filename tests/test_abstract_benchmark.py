import pytest

from hpobench.abstract_benchmark import AbstractBenchmark, AbstractMultiObjectiveBenchmark


def test_abstract_benchmark():
    with pytest.raises(NotImplementedError):
        AbstractBenchmark.get_configuration_space()

    with pytest.raises(NotImplementedError):
        AbstractBenchmark.get_fidelity_space()

    with pytest.raises(NotImplementedError):
        AbstractBenchmark.get_meta_information()

    with pytest.raises(NotImplementedError):
        AbstractMultiObjectiveBenchmark.get_objective_names()
