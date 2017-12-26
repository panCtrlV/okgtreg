#!/usr/bin/env bash

python -u script.py 1 > script-a-1.out
python -u script.py 2 > script-a-2.out
python -u script.py 3 > script-a-3.out
python -u script.py 4 > script-a-4.out
python -u script.py 5 > script-a-5.out
python -u script.py 6 > script-a-6.out
python -u script.py 7 > script-a-7.out
python -u script.py 8 > script-a-8.out
python -u script.py 9 > script-a-9.out
python -u script.py 10 > script-a-10.out

# nohup parallel -j 10 < run_script.sh &
