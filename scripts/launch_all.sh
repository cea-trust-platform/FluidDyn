#! /bin/bash

cp ../etudes/*ipynb .
source ../venv-flu1d/bin/activate
rm launch.log
for nb in *ipynb
do
    echo $nb
    sed -e 's/%matplotlib .*\\n/%matplotlib inline\\n/g'\
        -e 's/%matplotlib .*"/%matplotlib inline\\n"/g'\
        -e 's/save_fig[ =]*True/save_fig = False/g'\ -i $nb
    echo 'sed done, inline plot and no save_fig'

    # -e 's/[^ "]*savefig(.*"/"/g'\
    # -e 's/[^ "]*savefig(.*\\n/\\n/g'    echo 'sed done'
    jupyter-nbconvert --to notebook --execute $nb --inplace >> launch.log 2>&1 &
done


# -e 's/n_lim = .*\\n/n_lim = 1\\n/g'\
# -e 's/n_lim = .*"/n_lim = 1"/g'\
# -e 's/n_max = .*\\n/n_max = 1\\n/g'\
# -e 's/n_max = .*"/n_max = 1"/g'\
