#!/usr/bin/env bash

# Reference: http://pace.gatech.edu/submitting-multiple-jobs-quickly?destination=node%2F1621

for RESPONSE_ID in `seq 1 18`; do
    qsub -l procs=1 -l walltime=10:00:00 -v RESPONSE_ID=$RESPONSE_ID /home/panc/research/OKGT/software/okgtreg/okgtreg/application/crime_unnormalize_02262016/crime_unnormalize_screening_radon_sub.sh
done


# qselect -u <username> | xargs qdel
