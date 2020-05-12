#!/usr/bin/env sh

cd examples

for script in *.py
do
    python $script --on_travis
    rval=$?
    if [ "$rval" != 0 ]; then
        echo "Error running example $script"
        exit $rval
    fi
done