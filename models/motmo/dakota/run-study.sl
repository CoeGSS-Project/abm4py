#!/bin/bash
#
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -p standard

module load hdf5
module unload gcc
module load python/3.4.8
module load hdf5/1.10.1_openmpi-3.0.1_gcc620
module load lapack


time dakota -i sampling.in -o sampling.out > sampling.output
