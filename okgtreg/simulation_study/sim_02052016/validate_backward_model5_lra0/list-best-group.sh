#!/usr/bin/env bash

for FILE in validate_backward_lra0_radon-sub.sh.o*;
do
    #echo `grep 'Best group structure:' $(file)`
    grep 'Best group structure:' ./$FILE
done
