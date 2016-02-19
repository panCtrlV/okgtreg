#!/usr/bin/env bash

# Reference: http://pace.gatech.edu/submitting-multiple-jobs-quickly?destination=node%2F1621

for MODEL_ID in 2; do
	for DATA_SEED in 1; do
	    for BT_SEED in `seq 1 100`; do
            qsub -l procs=1 -l walltime=12:00:00 -v MODEL_ID=$MODEL_ID,DATA_SEED=$DATA_SEED,BT_SEED=$BT_SEED /home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016/validate_exhaustive_bootstrap_radon_sub.sh
        done
	done
done


# qselect -u <username> | xargs qdel

# DO NOT add space between the command line parameters