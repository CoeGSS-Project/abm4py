#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:06:54 2017

@author: geiges, Global climate forum
"""

from para_class_mobilityABM import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell,  Earth, Opinion
from para_lib_gcfabm import Location, GhostLocation
import bunch
import mpi4py.MPI as MPI
import class_auxiliary as aux
import numpy as np
# to be run with 4 nodes

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()


debug = False

parameters = bunch.Bunch()
parameters.nSteps = 5
parameters.isSpatial = 1
parameters.startDate = [01,2005]
parameters.timeUint  = 0
parameters.burnIn    = 0
parameters.omniscientBurnIn = 0
if mpiSize == 1:
    parameters.landLayer = np.asarray([0,0,0,0])
else:
    parameters.landLayer = np.asarray([0,1,2,3])
parameters.landLayer.shape = [4,1]


def testSpatialNodeCreation(comm, parameters):
    earth = Earth(parameters, 10, debug=debug, mpiComm=comm)
    
    # init type
    _cell    = earth.registerNodeType('cell' , AgentClass=Location, GhostAgentClass= GhostLocation, 
                                  propertyList = ['type', 
                                                  'gID',
                                                  'pos'])
    if not debug:
        assert _cell == 1
                                                  
    # init spatial layer 
    
    connList= aux.computeConnectionList(1, ownWeight=1)
    earth.initSpatialLayer(parameters.landLayer, connList, _cell, LocClassObject=Location, GhstLocClassObject=GhostLocation)  
    
    print earth.graph.vcount()
    if not debug:
        
        if mpiRank in [0,3]:
            assert earth.graph.vcount() == 2
        if mpiRank in [1,2]:
            assert earth.graph.vcount() == 3
        
    print 'test 1 done'
    return earth

def recordSyncTest(earth):
    
    earth.glob.registerValue('testSum', mpiRank, 'sum')
    earth.glob.registerStat('testMean', mpiRank, 'mean')
    earth.glob.sync()
    if not debug:
        assert earth.glob['testSum'] == 6
    
    assert earth.glob['testMean'] == 1.5
    
    print 'test 2 done'
    
earth = testSpatialNodeCreation(comm, parameters)    
recordSyncTest(earth)

