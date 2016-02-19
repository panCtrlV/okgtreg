#!/usr/bin/env bash

# Reference: http://pace.gatech.edu/submitting-multiple-jobs-quickly?destination=node%2F1621

for MODEL_ID in 5; do
	for SEED_NUM in `seq 1 100`; do
 		 qsub -l procs=1 -l walltime=10:00:00 -v MODEL_ID=$MODEL_ID,SEED_NUM=$SEED_NUM /home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016/validate_exhaustive_radon_sub.sh
	done
done


# qselect -u <username> | xargs qdel

