#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:42:57 2017

@author: gcf
"""



import sys
sys.path = ['/home/gcf/python/synEarth/agModel/h5py/build/lib.linux-x86_64-2.7'] + sys.path 
import mpi4py 
from mpi4py import MPI
print mpi4py.__file__
import h5py
#h5py.run_tests()


rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)

dset = f.create_dataset('test', (4,2), dtype='i')
dset[rank,] = rank, rank

f.close()

print mpi4py.__file__