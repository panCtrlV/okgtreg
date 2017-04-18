#!/usr/bin/env bash

# Reference: http://pace.gatech.edu/submitting-multiple-jobs-quickly?destination=node%2F1621

for MU_ID in `seq 1 5`; do
	for ALPHA_ID in `seq 1 10`; do
 		 qsub -l procs=1 -l walltime=40:00:00 -v MU_ID=$MU_ID,ALPHA_ID=$ALPHA_ID /home/panc/research/OKGT/software/okgtreg/okgtreg/application/crime_unnormalize_02262016/crime_unnormalize_murders_backward-radon-sub.sh
	done
done


# qselect -u <username> | xargs qdel
