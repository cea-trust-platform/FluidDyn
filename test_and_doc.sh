#!/bin/bash

mkdir -p doc/htmlcov
cd ./tests
pytest --nbval --cov=../src/ . --cov-report=html
for file in htmlcov/*
do
    if [[ -f $file ]]; then
        echo "$file"
        name=${file##*\/}
        cp $file ../doc/_build/html/doc/htmlcov/
    fi
done
echo -e "Test coverage\n=============\n\n\`test_coverage <./index.html>\`_\n\n" > ../doc/htmlcov/index_test.rst

cd ..
make html
