#!/usr/bin/env bash

#!/bin/bash
#
#SBATCH -n 28
#SBATCH --time=04:00:00
#SBATCH -p standard

module load hdf5
module unload gcc
module load python/3.4.8
module load hdf5/1.10.1_openmpi-3.0.1_gcc620
module load lapack

mpirun -np 28 -mca mpi_warn_on_fork 0 python3 ../run-via-dakota_prod.py $1 $2
