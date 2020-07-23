#!/usr/bin/env sh

if [[ "$RUN_CODESTYLE" == "true" ]]; then
    echo "Performing codestyle checking"

    test_codestyle=$(pycodestyle --max-line-length=120 ./hpolib)
    if [[ $test_codestyle ]]; then
      echo $test_codestyle
      exit 1
    else
      echo "Codesytle: No errors found"
    fi

    test_flake=$(flake8 --max-line-length=120 ./hpolib)
    if [[ $test_flake ]]; then
      echo $test_flake
      exit 1
    else
      echo "Flake8: No errors found"
    fi
else
    echo "Skip code style checking"
fi
