#!/bin/bash

RANK=$PMI_RANK

python -m cProfile -o output/scaling_$RANK.prof scaling_test.py
snakeviz scaling_0.prof
