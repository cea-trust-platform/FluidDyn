#!/bin/bash

cd ~/PycharmProjects/FluidDyn1D/
rsync --progress -auvz ./etudes/References/ titania:/home/catB/$USER/FluidDynBDD/
