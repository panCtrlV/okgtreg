#!/usr/bin/env bash

python -u script.py 2 > script-model-2.out
python -u script.py 5 > script-model-5.out
python -u script.py 6 > script-model-6.out
python -u script.py 7 > script-model-7.out

# nohup parallel -j 4 < run_script.sh &