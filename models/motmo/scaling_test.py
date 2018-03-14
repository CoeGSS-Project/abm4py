#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:15:50 2018

@author: ageiges
"""


import mpi4py
mpi4py.rc.threads = False
import sys, os
import socket
import time
#import guppy
from copy import copy
from os.path import expanduser

home = expanduser("~")

sys.path.append('../../lib/')
sys.path.append('../../modules/')



dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = ['../../h5py/build/lib.linux-x86_64-2.7'] + sys.path
    sys.path = ['../../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path

else:
    
    import matplotlib
    matplotlib.use('Agg')


import numpy as np

import lib_gcfabm as LIB #, GhostAgent, World,  h5py, MPI
import class_auxiliary as aux

#comm = abm.MPI.COMM_WORLD
#mpiRank = comm.Get_rank()
#mpiSize = comm.Get_size()

import logging as lg
import matplotlib.pylab as plt
import seaborn as sns; sns.set()
sns.set_color_codes("dark")
#import matplotlib.pyplot as plt

import pandas as pd
from bunch import Bunch

from scipy import signal

print 'import done'

#%% get mpi comm
mpiComm = LIB.MPI.COMM_WORLD
mpiRank = mpiComm.Get_rank()
mpiSize = mpiComm.Get_size()

#%% CONSTANTS - ENUMERATIONS
#CELL  = 1
#AGENT = 2

CON_CC = 1 # cell-to-cell onnnection
CON_AC = 2 # cell-to-agent connection
CON_AA = 3 # agent-to-agent connection

debug = 0
showFigures    = 0

minAgentPerCell = 2
maxAgentPerCell = 11

nFriends = 50
factor = 20
nSteps = 1000
radius = 3

baseOutputPath = aux.getEnvironment(mpiComm, getSimNo=False)
simNo = 0
outputPath = aux.createOutputDirectory(mpiComm, baseOutputPath, simNo)

#%% Setup of log file
if debug:
    lg.basicConfig(filename=outputPath + '/log_R' + str(mpiRank),
                filemode='w',
                format='%(levelname)7s %(asctime)s : %(message)s',
                datefmt='%m/%d/%y-%H:%M:%S',
                level=lg.DEBUG)
else:
    lg.basicConfig(filename=outputPath + '/log_R' + str(mpiRank),
                    filemode='w',
                    format='%(levelname)7s %(asctime)s : %(message)s',
                    datefmt='%m/%d/%y-%H:%M:%S',
                    level=lg.INFO)
mpiComm.Barrier()
if mpiComm.rank == 0:
    print 'log files created'
        
lg.info('Log file of process '+ str(mpiRank) + ' of ' + str(mpiSize))

parameters = dict()
parameters['nSteps'] = nSteps
parameters['showFigures'] = 0

#%% Init of world
earth = LIB.World(simNo,
              outputPath,
              True,
              parameters['nSteps'],
              maxNodes=1e6,
              debug=debug,
              mpiComm=mpiComm,
              caching=True,
              queuing=True)

earth.setParameters(parameters)
#%% Init of entity types
CELL    = earth.registerNodeType('cell' , AgentClass=LIB.Location, GhostAgentClass= LIB.GhostLocation,
                                     staticProperies  = ['type',
                                                       'gID',
                                                       'pos'],
                                     dynamicProperies = ['agentsPerCell'])

AGENT   = earth.registerNodeType('cell' , AgentClass=LIB.Agent, GhostAgentClass= LIB.GhostAgent,
                                     staticProperies  = ['type',
                                                   'gID',
                                                   'pos'],
                                     dynamicProperies =  ['prop_A',
                                                          'prop_B'])
#%% Init of edge types
earth.registerEdgeType('cell-cell', CELL, CELL, ['type','weig'])
earth.registerEdgeType('cell-ag', CELL, AGENT)
earth.registerEdgeType('ag-ag', AGENT,AGENT)

parameters['connRadius'] = radius

connList= aux.computeConnectionList(parameters['connRadius'], ownWeight=1.5)

procPerDim = int(np.sqrt(mpiSize))
layerShape = [procPerDim*factor,  procPerDim*factor]
parameters['landLayer'] = np.zeros(layerShape)

iProcess = 0
for x in range(procPerDim):     
    for y in range(procPerDim):
        
        parameters['landLayer'][x*factor:(x+1)*factor,y*factor:(y+1)*factor] = iProcess
        iProcess +=1
        

earth.initSpatialLayer(parameters['landLayer'],
                           connList, CELL,
                           LocClassObject=LIB.Location,
                           GhstLocClassObject=LIB.GhostLocation)

for cell in earth.iterEntRandom(CELL):
    cell.setValue('agentsPerCell', np.random.randint(minAgentPerCell,maxAgentPerCell))
    cell.peList = list()
earth.mpi.updateGhostNodes([CELL],['agentsPerCell'])


for cell in earth.iterEntRandom(CELL, ghosts=True):
    cell.peList = list()
    
    
# creation of agents
locDict = earth.getLocationDict()
for x, y in locDict.keys():
    loc         = earth.getEntity(locDict[x, y].nID)
    nAgentsCell = loc.getValue('agentsPerCell')
    
    for iAgent in range(nAgentsCell):
        agent = LIB.Agent(earth,
                      pos=(x, y),
                      prop_A = float(iAgent),
                      prop_B = np.random.random())
        agent.register(earth, parentEntity=loc, edgeType=CON_AC)
        agent.loc.peList.append(agent.nID)

if earth.queuing:
    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges(earth)

earth.mpi.transferGhostNodes(earth) 

if earth.queuing:
    earth.queue.dequeueVertices(earth)
    earth.queue.dequeueEdges(earth)   
    
for ghostCell in earth.iterEntRandom(CELL, ghosts = True, random=False):
    ghostCell.updatePeList(earth.graph, AGENT)
    

    
#earth.view('test2.png')
    
#%% connectin agents
globalEdgeList = list()
#globalWeightList = list()
for agent in earth.iterEntRandom(AGENT):
    contactList, connList, weigList = agent.getNClosePeers(earth, 
                                                           nFriends, 
                                                           edgeType=CON_AC,
                                                           addYourself=False)
    #connList = [(agent.nID, peerID) for peerID in np.random.choice(earth.nodeDict[AGENT],nFriends)]
    globalEdgeList.extend(connList)

earth.addEdges(globalEdgeList, type=CON_AA)
if earth.queuing:
    earth.queue.dequeueEdges(earth)    
#%% register of global records
earth.registerRecord('average_prop_B',
                     'sumation test for agents',
                     ['global sum of prob_B'])
earth.graph.glob.registerStat('mean_prop_B' , np.asarray([0]),'mean')
        
#%% init times
earth.compTime = np.zeros(nSteps)
earth.syncTime    = np.zeros(nSteps)
earth.waitTime    = np.zeros(nSteps)
earth.ioTime      = np.zeros(nSteps)
#%% init agent file 
earth.io.initNodeFile(earth, [CELL, AGENT])
#earth.io.gatherNodeData(0)
#earth.io.writeDataToFile(0)

def step(earth):
    
    
    tt = time.time()    
    for agent in earth.iterEntRandom(AGENT):
        
        peerValues = np.asarray(agent.getPeerValues('prop_B',CON_AA)[0])
        peerAverage = np.sum(peerValues / len(peerValues))
        #print peerAverage
        if peerAverage < 0.5:
            
            agent.setValue('prop_B', np.random.random())
    earth.compTime[earth.timeStep] += time.time()-tt
    
    
    tt = time.time()
    earth.mpi.updateGhostNodes([AGENT],['prop_B'])
    earth.syncTime[earth.timeStep] += time.time()-tt
    
    tt = time.time()
    earth.io.gatherNodeData(earth.timeStep)
    earth.io.writeDataToFile(earth.timeStep)
    earth.ioTime[earth.timeStep] += time.time()-tt
    
    tt = time.time()
    earth.mpi.comm.Barrier()
    earth.waitTime[earth.timeStep] += time.time()-tt

    tt = time.time()
    #earth.graph.glob.updateLocalValues('sum_prop_B', earth.getNodeValues('prop_B',AGENT))
    earth.graph.glob.updateLocalValues('mean_prop_B', np.asarray(earth.graph.vs[earth.nodeDict[AGENT]]['prop_B']))
        
    earth.graph.glob.sync()
    earth.globalRecord['average_prop_B'].set(earth.timeStep, earth.graph.glob['mean_prop_B'])
    earth.syncTime[earth.timeStep] += time.time()-tt
    
    
    if mpiRank == 0:
        
        print((str(earth.timeStep) + ' - Times: tComp: '+ '{:10.5f}'.format(earth.compTime[earth.timeStep])+
                  ' - tSync: '+ '{:10.5f}'.format(earth.syncTime[earth.timeStep])+
                  ' - tWait: '+ '{:10.5f}'.format(earth.waitTime[earth.timeStep])+
                  ' - tIO: '+ '{:10.5f}'.format(earth.ioTime[earth.timeStep]) ))
    earth.timeStep +=1
    
times = np.zeros(parameters['nSteps'])    

for iStep in range(parameters['nSteps']):
    tt = time.time()
    step(earth)
    times[iStep] =  time.time() -tt
print 'average time per step: ' + str(times.mean())

earth.io.finalizeAgentFile()
earth.finalize()

if mpiRank==0:
    plt.figure('average_prop_B')
    plt.plot(earth.globalRecord['average_prop_B'].rec)
    plt.savefig('test.png')
    
    plt.figure('times')
    plt.plot(times)
    plt.savefig('times.png')