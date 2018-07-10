#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:15:50 2018

@author: ageiges
"""


# S E T U P #################################################

minAgentPerCell = 5  # minimal number of agents per cell
maxAgentPerCell = 15 # minimal number of agetns per cell

nFriends = 50        # number of interconnections per agents -> increase computation
factor = 25          # spatial extend per process (factor x factor)
nSteps = 10          # number of model steps that are run
radius = 5           # spatial interaction radius -> increases of communication
weakScaling = True  # testin weak scaling

debug = True
if weakScaling == False:
    layerShape = [8, 8]

#############################################################





#import sys, mpi4py
#sys_excepthook = sys.excepthook
#def mpi_excepthook(v, t, tb):
#    sys_excepthook(v, t, tb)
#    mpi4py.MPI.COMM_WORLD.Abort(1)
import mpi4py
import sys, os
import socket
import time

from os.path import expanduser

home = expanduser("~")

sys.path.append('../lib/')
#sys.path = ['../h5py/build/lib.linux-x86_64-2.7'] + sys.path


#dir_path = os.path.dirname(os.path.realpath(__file__))
#if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
#    sys.path = ['../h5py/build/lib.linux-x86_64-2.7'] + sys.path
#    sys.path = ['../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path
#
#else:
#    
#    import matplotlib
#    matplotlib.use('Agg')


import numpy as np

import lib_gcfabm as LIB #, GhostAgent, World,  h5py, MPI
import core

import logging as lg
import matplotlib.pylab as plt

print('import done')



#%% get mpi comm
mpiComm = core.MPI.COMM_WORLD
mpiRank = mpiComm.Get_rank()
mpiSize = mpiComm.Get_size()

showFigures = 0

simNo, outputPath = core.setupSimulationEnvironment(mpiComm)


if not os.path.isfile(outputPath + '/run_times.csv'):
    fid = open(outputPath + '/run_times.csv','w')
    fid.write(', '.join([number  for number in ['mpiSize', 'tInit', 'tComp', 'tSync', 'tWait', 'tIO', 'tAveragePerStep', 'tOverall']]) + '\n')
    fid.close()

  
#%% Setup of log file
if mpiComm.size > 1:
    core.configureLogging(outputPath, debug=False)
    core.configureSTD(outputPath)
    
    
if mpiRank == 0:
    print('log files created')
        
lg.info('Log file of process '+ str(mpiRank) + ' of ' + str(mpiSize))

parameters = dict()
parameters['nSteps'] = nSteps
parameters['showFigures'] = 0

mpiComm.Barrier()

ttInit = time.time()
#%% Init of world
earth = LIB.World(simNo,
              outputPath,
              spatial=True,
              nSteps=parameters['nSteps'],
              maxNodes=1e5,
              maxEdges=1e7,
              debug=debug,
              mpiComm=mpiComm)

earth.setParameters(parameters)


#%% Init of entity types
CELL    = earth.registerAgentType('cell' , AgentClass=LIB.Location, GhostAgentClass= LIB.GhostLocation,
                                     staticProperties  = [('gID', np.int32, 1),
                                                         ('pos', np.int16, 2)],
                                     dynamicProperties = [('agentsPerCell', np.int16, 1)])

AGENT   = earth.registerAgentType('agent' , AgentClass=LIB.Agent, GhostAgentClass= LIB.GhostAgent,
                                     staticProperties  = [('gID', np.int32),
                                                         ('pos', np.int16, 2)],
                                     dynamicProperties = [('prop_A'),
                                                         ('prop_B', np.float32, 1)])

#%% Init of edge types
CON_CC = earth.registerLinkType('cell-cell', CELL, CELL, [('weig', np.float32, 1)])
CON_AC = earth.registerLinkType('cell-ag', CELL, AGENT)
CON_AA = earth.registerLinkType('ag-ag', AGENT,AGENT)

parameters['connRadius'] = radius


if weakScaling:
    procPerDim = int(np.sqrt(mpiSize))
    layerShape = [procPerDim*factor,  procPerDim*factor]
    parameters['landLayer'] = np.zeros(layerShape)

else:
    procPerDim = int(np.sqrt(mpiSize))
    factor = int(layerShape[0] / procPerDim)
    parameters['landLayer'] = np.zeros(layerShape)
    
iProcess = 0
for x in range(procPerDim):     
    for y in range(procPerDim):
        print(x, y)
        print(factor)
        parameters['landLayer'][x*factor:(x+1)*factor,y*factor:(y+1)*factor] = iProcess
        iProcess +=1    
    
connList= core.computeConnectionList(parameters['connRadius'], ownWeight=1.5)

earth.spatial.initSpatialLayer(parameters['landLayer'],
                           connList, 
                           LocClassObject=LIB.Location)

for cell in earth.random.iterNodes(CELL):
    cell.set('agentsPerCell', np.random.randint(minAgentPerCell,maxAgentPerCell))
    cell.peList = list()
    
earth.papi.updateGhostNodes([CELL],['agentsPerCell'])
if mpiRank == 0:
    print('spatial layer initialized')

for cell in earth.random.iterNodes(CELL, ghosts=True):
    cell.peList = list()
    
    
# creation of agents
locDict = earth.getLocationDict()
for x, y in list(locDict.keys()):
    loc         = earth.getAgent(locDict[x, y].nID)
    nAgentsCell = loc.get('agentsPerCell')
    
    for iAgent in range(nAgentsCell):
        agent = LIB.Agent(earth,
                          pos=(x, y),
                          prop_A = float(iAgent),
                          prop_B = np.random.random())
        agent.register(earth, parentEntity=loc, liTypeID=CON_AC)
#        agent.loc.peList.append(agent.nID)


earth.papi.transferGhostNodes(earth) 

if mpiRank == 0:
    print('Agents created')
    
    
#%% connectin agents
globalSourceList = list()
globalTargetList = list()
#globalWeightList = list()
for agent in earth.random.iterNodes(AGENT):
    contactList, connList, weigList = earth.spatial.getNCloseEntities(agent=agent, 
                                                                      nContacts=nFriends, 
                                                                      agTypeID=AGENT,
                                                                      addYourself=False)
    globalSourceList.extend(connList[0])
    globalTargetList.extend(connList[1])
    
earth.addLinks(CON_AA, globalSourceList, globalTargetList)

del  globalSourceList, globalTargetList
   
if mpiRank == 0:
    print('Agents connections created')    
#%% register of global records
earth.papi.updateGhostNodes([AGENT],['prop_B'])

earth.registerRecord('average_prop_B',
                     'sumation test for agents',
                     ['global sum of prob_B'],)
                     #mpiReduce= 'mean')
earth.graph.glob.registerStat('average_prop_B' , np.asarray([0]), 'mean')
        
#%% init times
earth.compTime    = np.zeros(nSteps)
earth.syncTime    = np.zeros(nSteps)
earth.waitTime    = np.zeros(nSteps)
earth.ioTime      = np.zeros(nSteps)
#%% init agent file 
earth.io.initNodeFile(earth, [CELL, AGENT])


def stepFunction(earth):
    
    
    tt = time.time()    
    for agent in earth.random.iterNodes(AGENT):
        
        peerValues = np.asarray(agent.getPeerAttr('prop_B',CON_AA))
        peerAverage = np.sum(peerValues / len(peerValues))
        #print peerAverage
        if peerAverage < 0.5:
            
            agent.set('prop_B', np.random.random())
    
    earth.compTime[earth.timeStep] += time.time() - tt
    
    
    tt = time.time()
    earth.papi.updateGhostNodes([AGENT],['prop_B'])
    earth.syncTime[earth.timeStep] += time.time() - tt
    
    tt = time.time()
    earth.io.writeDataToFile(earth.timeStep, [CELL, AGENT])
    earth.ioTime[earth.timeStep] += time.time() - tt
    
    tt = time.time()
    earth.papi.comm.Barrier()
    earth.waitTime[earth.timeStep] += time.time()-tt

    tt = time.time()
    #earth.graph.glob.updateLocalValues('sum_prop_B', earth.getAgentAttr('prop_B',AGENT))
    earth.graph.glob.updateLocalValues('average_prop_B', earth.getAgentAttr('prop_B', agTypeID=AGENT))
        
    earth.graph.glob.sync()
    earth.globalRecord['average_prop_B'].set(earth.timeStep, earth.graph.glob.globalValue['average_prop_B'])
    earth.syncTime[earth.timeStep] += time.time()-tt
    
    
    if mpiRank == 0:
        
        print((str(earth.timeStep) + ' - Times: tComp: '+ '{:10.5f}'.format(earth.compTime[earth.timeStep])+
                  ' - tSync: '+ '{:10.5f}'.format(earth.syncTime[earth.timeStep])+
                  ' - tWait: '+ '{:10.5f}'.format(earth.waitTime[earth.timeStep])+
                  ' - tIO: '+ '{:10.5f}'.format(earth.ioTime[earth.timeStep]) ))
    earth.timeStep +=1
    
times = np.zeros(parameters['nSteps'])    


if mpiRank == 0:
    tInit = time.time() - ttInit
    print('Time for model initialization: ' + str(tInit) + ' s')




for iStep in range(parameters['nSteps']):
    tt = time.time()
    stepFunction(earth)
    times[iStep] =  time.time() -tt


earth.io.finalizeAgentFile()
earth.finalize()

if mpiRank==0: 
    tAveragePerStep = times.mean()
    print('average time per step: ' + str(tAveragePerStep))
    tOverall = time.time() - ttInit
    print('overall time: ' + str(tOverall))    
    
if mpiRank==0:
    plt.figure('average_prop_B')
    plt.plot(earth.globalRecord['average_prop_B'].rec)
    plt.savefig(outputPath + 'test.png')
    
    plt.figure('times')
    plt.plot(times)
    plt.savefig(outputPath + '/times.png')

gatherData= np.asarray([earth.compTime, earth.syncTime, earth.waitTime, earth.ioTime])
gatherData = np.asarray(mpiComm.gather(gatherData, root=0))


if mpiRank==0: 
    print(np.asarray(gatherData).shape)
    gatherData = gatherData.mean(axis=2).mean(axis=0)
    tComp = gatherData[0]
    tSync = gatherData[1]
    tWait = gatherData[2]
    tIO   = gatherData[3]
    fid = open(outputPath + '/run_times.csv','a')
    fid.write(', '.join(["{:10.6f}".format(number)  for number in [mpiSize, tInit, tComp, tSync, tWait, tIO, tAveragePerStep, tOverall]]) + '\n')
    fid.close()
    




if False:
    #%%
    # plotting    
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.clf()
    dfTimes = pd.read_csv('output/run_times.csv')
    nRuns = 4
    
    colList = ['r', 'g', 'b', 'm']
    for start in range(nRuns):
        idx = np.arange(start,dfTimes.shape[0],nRuns)
        plt.plot(dfTimes['mpiSize'].loc[idx], dfTimes[' tOverall'].loc[idx], colList[start])
    plt.xlabel('number of processes')
    plt.ylabel('overall time')
    
