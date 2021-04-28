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

    try:
        config = HPOBenchConfig()
    except PermissionError as err:
        # We now if the link is set to /var/lib/hpobench  that it is looking for the socket dir for the container
        # thus, it has set the _is_container - flag.
        assert str(err) == '[Errno 13] Permission denied: \'/var/lib/hpobench\''

    del os.environ['SINGULARITY_NAME']
    config = HPOBenchConfig()
    assert not config._is_container
