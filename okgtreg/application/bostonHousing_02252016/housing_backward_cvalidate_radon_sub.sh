#!/bin/sh -l
# FILENAME:  validate_exhaustive_radon_sub.sh

module load anaconda
cd $PBS_O_WORKDIR
unset DISPLAY

# This is the path on Radon
python -u /home/panc/research/OKGT/software/okgtreg/okgtreg/application/bostonHousing_02252016/housing_backward_cvalidate.py ${MU_ID} ${ALPHA_ID}
