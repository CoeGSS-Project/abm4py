#!/bin/bash
#SBATCH -n 140
#SBATCH --ntasks-per-node=28
#SBATCH --time=03:00:00
#SBATCH -p standard
#SBATCH --mem=100000

cd ~

#old
module load python/2.7.12
#new
#module load python/2.7.14
#module load hdf5/1.10.1_openmpi-2.1.2_gcc620


#export PKG_CONFIG_PATH=/opt/exp_soft/local/haswell/igraph/0.7.1/lib/pkgconfig/
#export HDF5_USE_FILE_LOCKING=FALSE

cd python/gcfabm/models/motmo

if [ $# -eq 0 ];
then 
echo "execute run with fixed parameter set 'parameters_ger.csv'"; 
mpirun -np 140 -mca mpi_warn_on_fork 0 python init_motmo_prod.py parameters_ger.csv
else 
echo "execute run with parameter set '$1'";
mpirun -np 320 -mca mpi_warn_on_fork 0 python -OO init_motmo.py ${1}
fi
