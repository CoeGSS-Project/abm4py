#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:10:00 2017

@author: gcf
"""

from mpi4py import MPI
import array
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import numpy as np

array_size = int(sys.argv[1])

print rank   
#print "Sending package0 does work"     

x = comm.allreduce(rank*2,MPI.SUM)
print 'sum rnK: ',x

if rank ==0:
    out = comm.alltoall([[],[rank]])
    print out
else:
    out = comm.alltoall([[rank],[]])
    print out
    

if rank == 0:

    data = np.random.randn(array_size).astype(np.float64)
    request = comm.Isend([data,data.size,MPI.DOUBLE],1,11)
    request.wait()

elif rank == 1:
   
    data = np.random.randn(array_size).astype(np.float64)
    print 'data on',rank
    print data[1:10]
    
    comm.Recv(data,source=0, tag=11)

    print data[1:10]

    print 'Success 1'
