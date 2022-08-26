#!/bin/bash

cur_path="$PWD"
dir="`dirname $0`"
if [ $dir != '.' ]
then
    dir="${dir:2:100}"
    dir="$cur_path"/$dir
else
    dir="$cur_path"
fi
export FLU1D_ROOT=$dir
echo 'FLU1D_ROOT = '$FLU1D_ROOT

source $FLU1D_ROOT/venv-flu1d/bin/activate

for script in $(ls $FLU1D_ROOT/scripts/*sh)
do
    source $script
done

