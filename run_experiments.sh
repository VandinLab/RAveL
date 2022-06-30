#!/bin/env bash
# source ci_venv/bin/activate

python ids_creator.py
python main.py -t 1 -s 100
python main.py -t 2 -s 100
python main.py -t 3 -s 100
python main.py -t 4 -s 100
python main.py -t 4 -s 5
python main.py -t 5 -s 100
python main.py -t 5 -s 5
python main.py -t 6 -s 100
python main.py -t 7
python main.py -t 8