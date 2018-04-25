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
weakScaling = False  # testin weak scaling
#############################################################




import mpi4py
mpi4py.rc.threads = False
import sys, os
import socket
import time

from os.path import expanduser

home = expanduser("~")

sys.path.append('../lib/')



dir_path = os.path.dirname(os.path.realpath(__file__))
if socket.gethostname() in ['gcf-VirtualBox', 'ThinkStation-D30']:
    sys.path = ['../h5py/build/lib.linux-x86_64-2.7'] + sys.path
    sys.path = ['../mpi4py/build/lib.linux-x86_64-2.7'] + sys.path

else:
    
    import matplotlib
    matplotlib.use('Agg')


import numpy as np

import lib_gcfabm as LIB #, GhostAgent, World,  h5py, MPI
import class_auxiliary as aux

import logging as lg
import matplotlib.pylab as plt

print 'import done'



#%% get mpi comm
mpiComm = LIB.MPI.COMM_WORLD
mpiRank = mpiComm.Get_rank()
mpiSize = mpiComm.Get_size()


debug       = 0
showFigures = 0

simNo       = 0
baseOutputPath = ''
outputPath = aux.createOutputDirectory(mpiComm, baseOutputPath, simNo)

if not os.path.isfile(outputPath + '/run_times.csv'):
    fid = open(outputPath + '/run_times.csv','w')
    fid.write(', '.join([number  for number in ['mpiSize', 'tInit', 'tComp', 'tSync', 'tWait', 'tIO', 'tAveragePerStep', 'tOverall']]) + '\n')
    fid.close()
    
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

if mpiRank == 0:
    print 'log files created'
        
lg.info('Log file of process '+ str(mpiRank) + ' of ' + str(mpiSize))

parameters = dict()
parameters['nSteps'] = nSteps
parameters['showFigures'] = 0


ttInit = time.time()
#%% Init of world
earth = LIB.World(simNo,
              outputPath,
              spatial=True,
              nSteps=parameters['nSteps'],
              maxNodes=1e8,
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
                                     dynamicProperies = ['prop_A',
                                                         'prop_B'])

#%% Init of edge types
CON_CC = earth.registerEdgeType('cell-cell', CELL, CELL, ['type','weig'])
CON_AC = earth.registerEdgeType('cell-ag', CELL, AGENT)
CON_AA = earth.registerEdgeType('ag-ag', AGENT,AGENT)

parameters['connRadius'] = radius



if weakScaling:
    procPerDim = int(np.sqrt(mpiSize))
    layerShape = [procPerDim*factor,  procPerDim*factor]
    parameters['landLayer'] = np.zeros(layerShape)



else:
    layerShape = [128, 128]
    
    procPerDim = int(np.sqrt(mpiSize))
    factor = layerShape[0] / procPerDim
    parameters['landLayer'] = np.zeros(layerShape)
    
iProcess = 0
for x in range(procPerDim):     
    for y in range(procPerDim):
        
        parameters['landLayer'][x*factor:(x+1)*factor,y*factor:(y+1)*factor] = iProcess
        iProcess +=1    
    
connList= aux.computeConnectionList(parameters['connRadius'], ownWeight=1.5)

earth.initSpatialLayer(parameters['landLayer'],
                           connList, 
                           CELL,
                           LocClassObject=LIB.Location,
                           GhstLocClassObject=LIB.GhostLocation)

for cell in earth.iterEntRandom(CELL, random=False):
    cell.setValue('agentsPerCell', np.random.randint(minAgentPerCell,maxAgentPerCell))
    cell.peList = list()
    
earth.mpi.updateGhostNodes([CELL],['agentsPerCell'])
if mpiRank == 0:
    print 'spatial layer initialized'

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
    
#for ghostCell in earth.iterEntRandom(CELL, ghosts = True, random=False):
#    ghostCell.updatePeList(earth.graph, AGENT)
for ghostAgent in earth.iterEntRandom(AGENT, ghosts = True, random=False)    :
    ghostAgent.loc.peList.append(ghostAgent.nID)
    
    
if mpiRank == 0:
    print 'Agents created'
    
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

del  globalEdgeList
   
if mpiRank == 0:
    print 'Agents connections created'    
#%% register of global records

earth.registerRecord('average_prop_B',
                     'sumation test for agents',
                     ['global sum of prob_B'],)
                     #mpiReduce= 'mean')
earth.graph.glob.registerStat('average_prop_B' , np.asarray([0]), 'mean')
        
#%% init times
earth.compTime = np.zeros(nSteps)
earth.syncTime    = np.zeros(nSteps)
earth.waitTime    = np.zeros(nSteps)
earth.ioTime      = np.zeros(nSteps)
#%% init agent file 
earth.io.initNodeFile(earth, [CELL, AGENT])
#earth.io.gatherNodeData(0)
#earth.io.writeDataToFile(0)
#from tqdm import tqdm
def stepFunction(earth):
    
    
    tt = time.time()    
    for agent in earth.iterEntRandom(AGENT):
        
        peerValues = np.asarray(agent.getPeerValues('prop_B',CON_AA)[0])
        peerAverage = np.sum(peerValues / len(peerValues))
        #print peerAverage
        if peerAverage < 0.5:
            
            agent.setValue('prop_B', np.random.random())
    earth.compTime[earth.timeStep] += time.time() - tt
    
    
    tt = time.time()
    earth.mpi.updateGhostNodes([AGENT],['prop_B'])
    earth.syncTime[earth.timeStep] += time.time() - tt
    
    tt = time.time()
    earth.io.writeDataToFile(earth.timeStep)
    earth.ioTime[earth.timeStep] += time.time() - tt
    
    tt = time.time()
    earth.mpi.comm.Barrier()
    earth.waitTime[earth.timeStep] += time.time()-tt

    tt = time.time()
    #earth.graph.glob.updateLocalValues('sum_prop_B', earth.getNodeValues('prop_B',AGENT))
    earth.graph.glob.updateLocalValues('average_prop_B', np.asarray(earth.graph.vs[earth.nodeDict[AGENT]]['prop_B']))
        
    earth.graph.glob.sync()
    earth.globalRecord['average_prop_B'].set(earth.timeStep, earth.graph.glob['average_prop_B'])
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
    print 'Time for model initialization: ' + str(tInit) + ' s'




for iStep in range(parameters['nSteps']):
    tt = time.time()
    stepFunction(earth)
    times[iStep] =  time.time() -tt


earth.io.finalizeAgentFile()
earth.finalize()

if mpiRank==0: 
    tAveragePerStep = times.mean()
    print 'average time per step: ' + str(tAveragePerStep)
    tOverall = time.time() - ttInit
    print 'overall time: ' + str(tOverall)    
    
if mpiRank==0:
    plt.figure('average_prop_B')
    plt.plot(earth.globalRecord['average_prop_B'].rec)
    plt.savefig(outputPath + '/test.png')
    
    plt.figure('times')
    plt.plot(times)
    plt.savefig(outputPath + '/times.png')

gatherData= np.asarray([earth.compTime, earth.syncTime, earth.waitTime, earth.ioTime])
gatherData = np.asarray(mpiComm.gather(gatherData, root=0))


if mpiRank==0: 
    print np.asarray(gatherData).shape
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
    dfTimes = pd.read_csv('run_times.csv')
    nRuns = 4
    
    colList = ['r', 'g', 'b', 'm']
    for start in range(nRuns):
        idx = np.arange(start,dfTimes.shape[0],nRuns)
        plt.plot(dfTimes['mpiSize'].loc[idx], dfTimes[' tOverall'].loc[idx], colList[start])
    plt.xlabel('number of processes')
    plt.ylabel('overall time')
    
