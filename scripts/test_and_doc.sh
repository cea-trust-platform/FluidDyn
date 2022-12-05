#!/bin/bash

test_and_coverage(){
    [ -v FLU1D_ROOT ] || { echo "FLU1D_ROOT not set, env not sourced !"; exit; }
    cd $FLU1D_ROOT || exit
    mkdir -p doc/htmlcov
    cd ./tests
    python3 -m pytest --nbval --cov=../src/ --cov-config=.coveragerc . --cov-report=html
    for file in htmlcov/*
    do
        if [[ -f $file ]]; then
            echo "$file"
            # name=${file##*\/}
            \cp -f $file ../doc/_build/html/doc/htmlcov/
        fi
    done
    echo -e "Test coverage\n=============\n\n\`test_coverage <./index.html>\`_\n\n" > ../doc/htmlcov/index_test.rst
    cd ..
}
