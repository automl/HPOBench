import pytest

from hpolib.abstract_benchmark import AbstractBenchmark

with pytest.raises(NotImplementedError):
    AbstractBenchmark.get_configuration_space()

with pytest.raises(NotImplementedError):
    AbstractBenchmark.get_fidelity_space()

with pytest.raises(NotImplementedError):
    AbstractBenchmark.get_meta_information()
