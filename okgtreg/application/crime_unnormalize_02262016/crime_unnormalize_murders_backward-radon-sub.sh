#!/bin/sh -l
# FILENAME:  validate_exhaustive_radon_sub.sh

module load anaconda
cd $PBS_O_WORKDIR
unset DISPLAY

# This is the path on Radon
python -u /home/panc/research/OKGT/software/okgtreg/okgtreg/application/crime_unnormalize_02262016/crime_unnormalize_murders_backward.py ${MU_ID} ${ALPHA_ID}
