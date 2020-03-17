#!/usr/bin/env sh

if [[ "$RUN_CODESTYLE" == "true" ]]; then
    echo "Performing codestyle checking"
    pycodestyle --max-line-length=120 ./hpolib
    flake8 --max-line-length=120 ./hpolib
else
    echo "Skip code style checking"
fi
