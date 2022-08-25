#! /bin/bash

cd ../local/ || exit
rm ./*ipynb
cp ../etudes/*ipynb .
source ../venv-flu1d/bin/activate
logfile='launch.log'
rm -f $logfile
for nb in *ipynb
do
    echo $nb >> $logfile
    sed -e 's/%matplotlib .*\\n/%matplotlib inline\\n/g'\
        -e 's/%matplotlib .*"/%matplotlib inline\\n"/g'\
        -e 's/save_fig[ =]*True/save_fig = False/g' -i "$nb" 1>> $logfile 2>&1
    echo 'sed done, inline plot and no save_fig' >> $logfile

    # -e 's/[^ "]*savefig(.*"/"/g'\
    # -e 's/[^ "]*savefig(.*\\n/\\n/g'    echo 'sed done'
    jupyter-nbconvert --to notebook --execute "$nb" --inplace 1>> $logfile 2>&1 &
done


# -e 's/n_lim = .*\\n/n_lim = 1\\n/g'\
# -e 's/n_lim = .*"/n_lim = 1"/g'\
# -e 's/n_max = .*\\n/n_max = 1\\n/g'\
# -e 's/n_max = .*"/n_max = 1"/g'\
