#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:06:54 2017

@author: geiges, Global climate forum

execute with: mpiexec -n 4 python hdf5_parallel_test.py 
"""
import os
import socket
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = ['../h5py/build/lib.linux-x86_64-2.7'] + sys.path
    sys.path = ['../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path

    
    
#from classes_motmo import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell,  Earth, Opinion
#from lib_gcfabm import Location, GhostLocation
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

dset = f.create_dataset('test', (mpiSize,2), dtype='f')
#data = dset[mpiRank:mpiRank+1,]
#data[0,:] = mpiRank, np.random.randn()
dset[mpiRank,:] = mpiRank, np.random.randn()
comm.Barrier()

f.close()

if mpiRank == 0:  
    print 'h5py parallel write I/O successful'

h5File      = h5py.File('parallel_test.hdf5', 'r', driver='mpio', comm=comm)
dset = h5File.get('test')
data = dset[mpiRank:mpiRank+1,]
print data
assert data[0,0] == mpiRank

h5File.close()
if mpiRank == 0:  
    print 'h5py parallel read I/O successful'        

