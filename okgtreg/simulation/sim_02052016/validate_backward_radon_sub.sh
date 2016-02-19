#!/bin/sh -l
# FILENAME:  validate_backward_radon_sub.sh

module load anaconda
cd $PBS_O_WORKDIR
unset DISPLAY

# This is the path on Radon
python -u /home/panc/research/OKGT/software/okgtreg/okgtreg/simulation/sim_02052016/validate.py ${MODEL_ID} ${SEED_NUM}

# Once you have a job submission file, you may submit this
# script to PBS using the qsub command.

# Submit job to the queue
#qsub validate_backward_radon_sub.sh

# Submit job with specified wall time (1 hour 30 min)
#qsub -l walltime=01:30:00 validate_backward_radon_sub.sh

# Submit job with specified # nodes and # cores/node
#qsub -l nodes=2:ppn=4 validate_backward_radon_sub.sh

# Submit job with specified # cores (free placement of nodes)
#qsub -l procs=100 -l walltime=02:30:00 validate_backward_radon_sub.sh

