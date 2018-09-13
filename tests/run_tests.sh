#!/bin/bash


mpirun -np 4 python  mpiTest.py
mpirun -np 4 python  hdf5_parallel_test.py
mpirun -np 2 python  lib_test.py
mpirun -np 4 python  scaling_test.py
