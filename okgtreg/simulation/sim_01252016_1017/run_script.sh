#!/usr/bin/env bash

python -u script.py 1 > script-model-1.out
python -u script.py 2 > script-model-2.out
python -u script.py 3 > script-model-3.out
python -u script.py 4 > script-model-4.out
python -u script.py 5 > script-model-5.out
python -u script.py 6 > script-model-6.out
python -u script.py 7 > script-model-7.out
python -u script.py 8 > script-model-8.out
python -u script.py 9 > script-model-9.out
python -u script.py 10 > script-model-10.out

# nohup parallel -j 10 < run_script.sh &