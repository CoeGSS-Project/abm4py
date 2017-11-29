#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:06:54 2017

@author: geiges, Global climate forum

exec command: mpiexec -n 4 python para_run_tests.py
"""

from para_class_mobilityABM import Person, GhostPerson, Household, GhostHousehold, Reporter, Cell, GhostCell,  Earth, Opinion
from para_lib_gcfabm import Location, GhostLocation
import bunch
import mpi4py.MPI as MPI
import class_auxiliary as aux
import numpy as np
import h5py
import sys
# to be run with 4 nodes

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()



#olog_file  = open('output/log' + str(mpiRank) + '.txt', 'w')
#sys.stdout = olog_file
#elog_file  = open('output/err' + str(mpiRank) + '.txt', 'w')
#sys.stderr = elog_file

debug = False

parameters = bunch.Bunch()
parameters.nSteps = 5
parameters.isSpatial = 1
parameters.startDate = [01,2005]
parameters.timeUnit  = 0
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
                                                  'pos',
                                                  'prop'])
    if not debug:
        assert _cell == 1
                                                  
    # init spatial layer 
    
    connList= aux.computeConnectionList(1, ownWeight=1)
    earth.initSpatialLayer(parameters.landLayer, connList, _cell, LocClassObject=Location, GhstLocClassObject=GhostLocation)  
    earth.graph.vs['prop'] = 0
    earth.dprint(earth.graph.vcount())
    if not debug:
        
        if mpiRank in [0,3]:
            assert earth.graph.vcount() == 2
        if mpiRank in [1,2]:
            assert earth.graph.vcount() == 3
        
    print 'spatial creation test successful'
    return earth, _cell

def recordSyncTest(earth):
    
    earth.glob.registerValue('testSum', mpiRank, 'sum')
    earth.glob.registerStat('testMean', mpiRank, 'mean')
    earth.glob.sync()
    if not debug:
        assert earth.glob['testSum'] == 6
    
    if not debug:
        assert earth.glob['testMean'] == 1.5
    
    print 'MPI statistic sycn test successful'

def h5pyReadTest(earth):
    h5File      = h5py.File('resources_NBH/people1518.hdf5', 'r', driver='mpio', comm=earth.mpi.comm)
    dset = h5File.get('people')
    data = dset[(mpiRank)*10:(mpiRank+1)*10,]
    
    #print data
    earth.mpi.comm.Barrier()
    h5File.close()
    
    print 'h5py I/O test successful'
    
def communicationTest(earth):

    earth.dprint( (mpiRank, earth.graph.vs['prop']))
    earth.timeStep = 0
    
    def step(earth):
        
        earth.dprint('##### Time step ' + str(earth.timeStep) +' #############')

        
        earth.mpi.updateGhostNodes()
        
        earth.dprint('before update r', mpiRank, earth.graph.vs['prop'])
        for node in earth.iterEntRandom(_cell):
            props , __ = node.getPeerValues('prop',_cell)
            
            
            print node.getPeerIDs(_cell)
            assert props == props
            earth.dprint(str(node.nID) + ':' + str(props))
            if any(props):
                node.setValue('prop',np.max(props))
        
        
        
        
        if mpiRank == 0:
            earth.entDict[0].addValue('prop',1)
        earth.dprint('after update r',mpiRank, earth.graph.vs['prop'])
    for i in range(5):
        
        step(earth)
        earth.dprint(earth.entDict[0].node['prop'], max(0, earth.timeStep+1-mpiRank))
        assert earth.entDict[0].node['prop'] == max(0, earth.timeStep+1-mpiRank)
        earth.timeStep +=1
        
    print 'communication test successful'
        
    
def edgeReadValueTest(earth):
    _tcc = earth.registerEdgeType('cell-cell',_cell, _cell, ['type','weig'])
    
    for node in earth.iterEntRandom(_cell):
        props , __ = node.getEdgeValues('weig',_tcc)   
        print props
        print node.cache.edgesByType.keys()
    print "first round done"

    earth.mpi.comm.Barrier()
    print "second round"
    
    if mpiRank == 0:
        node = earth.entList[0]
        print node.cache.edgesByType
        node.addConnection(1, _tcc, weig=1)
    
    for node in earth.iterEntRandom(_cell):
        props , __ = node.getEdgeValues('weig',_tcc)   
        print props
earth , _cell = testSpatialNodeCreation(comm, parameters)    
recordSyncTest(earth)
h5pyReadTest(earth)
communicationTest(earth)
edgeReadValueTest(earth)