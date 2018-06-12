#!/bin/bash


mpirun -np 4 python -m mpi4py mpiTest.py
mpirun -np 4 python -m mpi4py hdf5_parallel_test.py
mpirun -np 2 python -m mpi4py lib_test.py
mpirun -np 4 python -m mpi4py scaling_test.py
