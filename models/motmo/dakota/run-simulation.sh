#!/usr/bin/env bash

if [ "$SYSTEM_NAME" == "workstation" ]
then
    python3 ../run-via-dakota.py $1 $2
elif [ "$SYSTEM_NAME" == "eagle" ]
then
   sbatch run-simulation.sl $1 $2
fi
