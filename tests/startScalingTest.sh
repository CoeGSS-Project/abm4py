#!/bin/bash
#SBATCH -n 256
#SBATCH --ntasks-per-node=28
#SBATCH --time=00:20:00
#SBATCH -p standard

cd ~

#old
#module load python/2.7.12
#new
module load python/2.7.14
module load hdf5/1.10.1_openmpi-2.1.2_gcc620


#export PKG_CONFIG_PATH=/opt/exp_soft/local/haswell/igraph/0.7.1/lib/pkgconfig/
#export HDF5_USE_FILE_LOCKING=FALSE

cd python/gcfabm/tests/

echo "execute scaling test'"; 
mpirun -np 256 -mca io romio314 -mca mpi_warn_on_fork 0 python scaling_test.py
