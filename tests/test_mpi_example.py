#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:10:00 2017

@author: gcf
"""



from mpi4py import MPI
import array

import pickle

def loadObj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


print rank   
print "Sending package0 does work"   


  
if rank == 0:
    data = loadObj('package0')
    #buf = array.array('b', 1000000 *[0]) 
    comm.send(data, dest=1, tag=11)
    #req = comm.irecv(buf=buf, source=1, tag=11)
    #data = req.wait()
    #print data
    
elif rank == 1:
    dataOrg = loadObj('package0')
    buf = array.array('b', 1000000 *[0]) 
    
    req = comm.irecv(buf=buf,source=0, tag=11)
    data = req.wait()
    #data = comm.recv(source=0,tag=11)
    assert data == dataOrg
    print 'Success 1'
#comm.Barrier()

print "Sending package0 does NOT work"     
if rank == 0:
    data = loadObj('package1')
    #buf2 = array.array('b', 1000000 *[0]) 
    comm.isend(data, dest=1, tag=12)
    #req = comm.irecv(buf=buf, source=1, tag=11)
    #data = req.wait()
    #print data
    
elif rank == 1:
    dataOrg = loadObj('package1')
    buf2 = array.array('b', 1000000 *[0]) 
    comm.isend(data, dest=1, tag=11)
    req = comm.irecv(buf=buf2,source=0, tag=12)
    data = req.wait()
    #data = comm.recv(source=0,tag=12)
    assert data == dataOrg
    print 'Success 2'