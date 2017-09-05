#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:42:57 2017

@author: gcf
"""


import os.path
import sys

#dir_path = os.path.dirname(os.path.realpath(__file__))
#parent_dir = dir_path[:dir_path.rfind(os.path.sep)]
#
#import sys
#sys.path = [parent_dir + '/h5py/build/lib.linux-x86_64-2.7'] + sys.path 

from mpi4py import MPI
import numpy as np
import h5py

#h5py.run_tests()

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
size =  MPI.COMM_WORLD.size

f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=comm)

dset = f.create_dataset('test', (size,2), dtype='f')
dset[rank,] = rank, np.random.randn()

f.close()

print mpi4py.__file__