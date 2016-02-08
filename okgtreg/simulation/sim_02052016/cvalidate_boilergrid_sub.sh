#!/usr/bin/env bash

# FILENAME: cvalidate_boilergrid_sub.sh

universe            = standard
transfer_executable = TRUE
executable          = cvalidate.py
arguments           = ${MODEL_ID} ${SEED_NUM}

# standard I/O files, HTCondor log file
output  = "cvalidate-model${MODEL_ID}-seed${SEED_NUM}.out"
error   = "cvalidate-model${MODEL_ID}-seed${SEED_NUM}.err"
log     = "cvalidate-model${MODEL_ID}-seed${SEED_NUM}.log"

# queue one job
queue