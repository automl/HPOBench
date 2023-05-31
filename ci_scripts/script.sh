#!/usr/bin/env sh

if [[ "$RUN_TESTS" == "true" ]]; then
    if [[ "$RUN_CODECOV" == "true" ]]; then
        echo "Run tests with code coverage"
        pytest -sv --cov=hpobench tests/
        exit_code=$?

        echo "Run code coverage"
        codecov
    else
        echo "Run tests without code coverage"
        pytest -sv tests/
        exit_code=$?
    fi

    if [[ "$exit_code" -eq 0 ]]; then
        echo "All test have passed."
    else
        echo "Some Tests have failed."
        exit 1
    fi
fi

if [[ "$RUN_CODESTYLE" == "true" ]]; then
    echo "Run codestyle"
    chmod +x ci_scripts/codestyle.sh && source ./ci_scripts/codestyle.sh
fi

if [[ "$RUN_CONTAINER_EXAMPLES" == "true" ]]; then
    echo "Run containerized examples"
    chmod +x ci_scripts/container_examples.sh && source ./ci_scripts/container_examples.sh
fi

if [[ "$RUN_LOCAL_EXAMPLES" == "true" ]]; then
    echo "Run containerized examples"
    chmod +x ci_scripts/local_examples.sh && source ./ci_scripts/local_examples.sh
fi