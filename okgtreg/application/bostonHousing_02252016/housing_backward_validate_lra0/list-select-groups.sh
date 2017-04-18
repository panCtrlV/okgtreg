#!/usr/bin/env bash

for FILE in housing_backward_radon_sub.sh.o*;
do
    #echo `grep 'Best group structure:' $(file)`
    grep 'Selected group structure:' ./$FILE
    grep 'Test error: ' ./$FILE
done


# Reference: Regular Expressions for file name matching
# http://stackoverflow.com/questions/4307770/regular-expressions-for-file-name-matching