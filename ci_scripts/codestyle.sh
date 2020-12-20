#!/usr/bin/env sh

if [[ "$RUN_CODESTYLE" == "true" ]]; then
    echo "Performing codestyle checking"

    test_codestyle=$(pycodestyle --max-line-length=120 ./hpobench)
    if [[ $test_codestyle ]]; then
      echo $test_codestyle
      exit 1
    else
      echo "Codesytle: No errors found"
    fi

    test_flake=$(flake8 --max-line-length=120 ./hpobench)
    if [[ $test_flake ]]; then
      echo $test_flake
      exit 1
    else
      echo "Flake8: No errors found"
    fi

    # Enable the error W0221: Parameters differ from overridden method (arguments-differ)
    test_pylint=$(pylint --disable=all --enable=W0221 ./hpobench)
    if [[ $test_pylint ]]; then
      echo $test_pylint
      exit 1
    else
      echo "Pylint: No signature errors found"
    fi

    # Just print the pylint output without throwing an error.
    pylint --exit-zero --rcfile=./pylint.rc ./hpolib

else
    echo "Skip code style checking"
fi
