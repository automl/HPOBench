from pathlib import Path


def test_config_file():
    from hpolib import config_file
    assert config_file.verbosity == 0
    assert config_file.config_version == '0.0.5'
    assert config_file.data_dir == Path('~/.local/share/hpolib2').expanduser().absolute()
    assert config_file.container_dir == Path('~/.cache/hpolib2/hpolib2-1000').expanduser().absolute()
    assert config_file.container_source == 'library://phmueller/automl'
    assert config_file.use_global_data
    assert config_file.pyro_connect_max_wait == 400
