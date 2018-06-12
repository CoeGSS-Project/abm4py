#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:06:54 2017

@author: geiges, Global climate forum

execute with: mpiexec -n 4 python hdf5_parallel_test.py 
"""

import mpi4py.MPI as MPI
#import class_auxiliary as aux
import numpy as np


import h5py

# to be run with 4 nodes

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

info = MPI.Info.Create()
info.Set("romio_ds_write", "disable")
info.Set("romio_ds_read", "disable")

#setup 1
f = h5py.File('parallel_test.hdf5', 'w', 
              driver='mpio', 
              comm=comm,)


#setup 2
#f = h5py.File('parallel_test.hdf5', 'w', 
#              driver='mpio', 
#              comm=comm,
#              libver='latest',
#              info = info)

dset = f.create_dataset('/test', (mpiSize,2), dtype='f')
#data = dset[mpiRank:mpiRank+1,]
#data[0,:] = mpiRank, np.random.randn()
dset[mpiRank,:] = mpiRank, np.random.randn()

attrStrings = [string.encode('utf8') for string in  ['test_str1', 'test_str2']]
dset.attrs.create('attribute_A', attrStrings)
comm.Barrier()

f.close()

if mpiRank == 0:  
    print('h5py parallel write I/O successful')

h5File      = h5py.File('parallel_test.hdf5', 'r', driver='mpio', comm=comm)
dset = h5File.get('/test')
data = dset[mpiRank:mpiRank+1,]
print(data)
assert data[0,0] == mpiRank

h5File.close()
if mpiRank == 0:  
    print('h5py parallel read I/O successful')        
    import os
    os.remove('parallel_test.hdf5')

comm.Barrier()

#structured array test
h5File = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=comm,)

dtype = np.dtype([('floatValue', np.float64, 2), ('rank', np.int32, 1)])
dset = h5File.create_dataset('/test', (mpiSize,), dtype=dtype)

testValues =np.random.randn(2)
dset[mpiRank,] = testValues, mpiRank
h5File.flush()
h5File.close()

h5File      = h5py.File('parallel_test.hdf5', 'r', driver='mpio', comm=comm)
dset = h5File.get('/test')
print(dset)
data = dset[mpiRank:mpiRank+1]
print(data)
assert data['rank'] == mpiRank

h5File.close()
comm.Barrier()
if mpiRank == 0:  
    print('h5py structured parallel write/read I/O successful')   