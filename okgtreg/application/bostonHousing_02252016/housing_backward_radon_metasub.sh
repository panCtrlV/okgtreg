#!/usr/bin/env bash

# Reference: http://pace.gatech.edu/submitting-multiple-jobs-quickly?destination=node%2F1621

for MU_ID in `seq 1 5`; do
	for ALPHA_ID in `seq 1 10`; do
 		 qsub -l procs=1 -l walltime=3:00:00 -v MU_ID=$MU_ID,ALPHA_ID=$ALPHA_ID /home/panc/research/OKGT/software/okgtreg/okgtreg/application/bostonHousing_02252016/housing_backward_radon_sub.sh
	done
done


# qselect -u <username> | xargs qdel
