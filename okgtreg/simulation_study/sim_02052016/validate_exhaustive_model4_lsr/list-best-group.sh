#!/usr/bin/env bash

for FILE in validate_exhaustive_radon_sub.sh.o*;
do
    #echo `grep 'Best group structure:' $(file)`
    grep 'Best group structure:' ./$FILE
done
