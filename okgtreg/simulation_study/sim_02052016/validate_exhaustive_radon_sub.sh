#!/bin/sh -l
# FILENAME:  validate_exhaustive_radon_sub.sh

module load anaconda
cd $PBS_O_WORKDIR
unset DISPLAY

# This is the path on Radon
python -u /home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016/validate_exhaustive.py ${MODEL_ID} ${SEED_NUM}