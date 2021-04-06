def test_version_check():
    from hpobench.config import HPOBenchConfig

    # Same partition
    assert HPOBenchConfig._check_version('0.0.0', '0.0.1')
    assert HPOBenchConfig._check_version('0.0.0', '0.0.5')

    assert HPOBenchConfig._check_version('0.0.6', '0.0.7')
    assert HPOBenchConfig._check_version('0.0.8', '0.0.1234')
    assert HPOBenchConfig._check_version('1.0.0', '0.0.8')

    assert not HPOBenchConfig._check_version(None, '0.0.1')
    assert not HPOBenchConfig._check_version(None, '0.0.6')
    assert not HPOBenchConfig._check_version('0.0.5', '0.0.6')


def test_is_container():
    import os
    os.environ['SINGULARITY_NAME'] = 'test_name'
    from hpobench.config import HPOBenchConfig

    config = HPOBenchConfig()
    assert config._is_container

    del os.environ['SINGULARITY_NAME']
    config = HPOBenchConfig()
    assert not config._is_container
