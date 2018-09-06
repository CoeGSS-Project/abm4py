#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://wwww.globalclimateforum.org

This example file is part on GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://earth.gnu.org/licenses/>.
"""
#%% load modules

import sys 
import os
import numpy as np
import logging as lg
import time
import random
import h5py

import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../../lib/')

import lib_abm4py_prod as LIB #, GhostAgent, World,  h5py, MPI
import core_prod as core
import tools
#%% CONFIG
N_AGENTS   = 1000
N_STEPS    = 1000
MAX_EXTEND = 50

IMITATION = 15
INNOVATION = .2

N_FRIENDS  = 10

REDUCTION_FACTOR = 100000

DEBUG = True
DO_PLOT  = True


BLUE = plt.get_cmap('Blues')(.3)
RED  = plt.get_cmap('RdPu_r')(.1)

#%% setup
simNo, outputPath = core.setupSimulationEnvironment()

world = LIB.World(simNo,
              outputPath,
              spatial=True,
              nSteps=N_STEPS,
              maxNodes=1e4,
              maxLinks=1e5,
              debug=DEBUG)

CELL = world.registerAgentType('cell' , AgentClass=LIB.Location,
                              staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.float32, 2),
                                                    ('imit', np.float16, 1),
                                                    ('nAgents', np.int16,1)],
                              dynamicProperties = [('fraction', np.int16, 1)])

AGENT = world.registerAgentType('agent' , AgentClass=LIB.Agent,
                               staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.float32, 2),
                                                    ('age',  np.int16, 1),
                                                    ('gender',  np.int16, 1),
                                                    ('income',  np.int32, 1),
                                                    ('nPers',  np.int16, 1),
                                                    ('imit', np.float16, 1),
                                                    ('inno', np.float16,1)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
#%% Init of edge types
LI_CC = world.registerLinkType('ce-ce', CELL, CELL, staticProperties =['weig'])
LI_CA = world.registerLinkType('ce-ag', CELL,AGENT)
LI_AA = world.registerLinkType('ag-ag', AGENT,AGENT)


#%% creating locations
populationMap = np.load('coarse_pop_count.npy')
populationMap = np.flipud(populationMap /REDUCTION_FACTOR).transpose()
populationMap.shape
IDArray = populationMap * np.nan
for x in range(populationMap.shape[0]):
    for y in range(populationMap.shape[1]):
        
        if not np.isnan(populationMap[x,y]):
            cell = LIB.Location(world,
                          pos=(x, y),
                          fraction=0,
                          nAgents=max(1,np.int(populationMap[x,y])))
            cell.register(world)
            world.registerLocation(cell, x, y)
            IDArray[x,y] = cell.nID

# %%create location network
connBluePrint = world.spatial.computeConnectionList(radius=2.5)
world.spatial.connectLocations(IDArray, connBluePrint, LI_CC)

tools.plotGraph(world, CELL, LI_CC)
#%%

h5File = h5py.File('italians.hdf5', 'r')
dset = h5File.get('people')
personData = dset[:,:5]
H5_NPERS  = 0
H5_AGE    = 1
H5_GENDER = 2
H5_INCOME = 3
H5_HHTYPE = 4

locDict = world.getLocationDict()
currIdx = 0


##############################################
# change the propertyToPreference funciton so 
# that is relies on the properties

def propertyToPreference(age, gender, income, hhType):
    
    inno = random.random() * INNOVATION
    imit = random.normalvariate(IMITATION,2 ) 
    
    return inno, imit

##############################################

for xLoc, yLoc in list(locDict.keys()):  
    loc = world.getNodeBy.location(xLoc, yLoc)
    
    
    for iAgent in range(loc.get('nAgents')):
        x = random.normalvariate(xLoc,.25)
        y = random.normalvariate(yLoc,.25)
        nPers   = int(personData[currIdx, H5_NPERS])
        
        age     = personData[currIdx, H5_AGE]
        gender  = personData[currIdx, H5_GENDER]
        income  = personData[currIdx, H5_INCOME]
        hhType  = personData[currIdx, H5_HHTYPE]
    
        inno, imit = propertyToPreference(age, gender, income, hhType)
        
        agent = LIB.Agent(world,
                          pos=(x, y),
                          switch = 0,
                          color = BLUE,
                          imit = imit,
                          inno = inno,
                          nPers =nPers,
                          age = age,
                          income = income,
                          gender = gender)
        
        agent.register(world)
        currIdx +=1

#%% creation of spatial proximity network
  
positions = world.getAttrOfAgentType('pos', agTypeID=AGENT)
agIDList  = world.getAgentIDs(AGENT)
innovationVal = world.getAttrOfAgentType('inno', agTypeID=AGENT).astype(np.float64)

for agent in world.getAgentsByType(AGENT):
    ##############################################
    # create a new creation rule 
    
    # spatial weight
    weig1 = np.sum((positions - agent.attr['pos'])**4,axis=1)
    weig1 = np.divide(1.,weig1, out=np.zeros_like(weig1), where=weig1!=0)
    
    # preference weight
    weig2 = np.abs((innovationVal - agent.attr['inno'])**2)
    weig2 = np.divide(1.,weig2, out=np.zeros_like(weig2), where=weig2!=0)
    
    # merging of weights
    weig = weig1 * weig2
    weig = weig / np.sum(weig)

    ##############################################
    
    
    friendIDs = np.random.choice(agIDList, N_FRIENDS, replace=False, p=weig)
    
    [agent.addLink(ID, liTypeID = LI_AA) for ID in friendIDs]
    

    
    
positions = world.getAttrOfAgentType('pos',agTypeID=AGENT)

##############################################
# exchange the position of spatial space (x,y) with the properties (inno, imit)

#positions[:,0] = 

##############################################

#%%
plt.figure('statistics')
plt.subplot(2,2,1)
data = world.getAttrOfAgentType('age',agTypeID=AGENT)
plt.hist(data)
plt.title('age distribution')
plt.subplot(2,2,2)
data = world.getAttrOfAgentType('income',agTypeID=AGENT)
plt.hist(data)
plt.title('income distribution')
plt.subplot(2,2,3)
data = world.getAttrOfAgentType('nPers',agTypeID=AGENT)
plt.hist(data)
plt.title('household size')
plt.subplot(2,2,4)
data = world.getAttrOfAgentType('nPers',agTypeID=AGENT)
plt.scatter(world.getAttrOfAgentType('income',agTypeID=AGENT), world.getAttrOfAgentType('age',agTypeID=AGENT))

plt.title('relation income to age')
plt.draw()

#%% Scheduler
iStep = 0
fracList = list()


if DO_PLOT:
    plotting = tools.PlotClass(positions, world, AGENT, LI_AA)
while True:
    tt =time.time()
    iStep+=1
    switched = world.getAttrOfAgentType('switch',agTypeID=AGENT)
    switchFraction = np.sum(switched) / world.nAgents(AGENT)
    fracList.append(switchFraction)
    
    
    if switchFraction == 1 or iStep == N_STEPS:
        break
    tools.printfractionExceed(switchFraction, iStep)
    
    nodesToIter = world.filterAgents(AGENT, 'switch', 'eq', 0)
    randValues  = np.random.random(len(nodesToIter))*1000
    
    for agent, randNum in zip(world.getAgentsByType(localIDs=nodesToIter),randValues) :
        
        # dynamic of the agent
        switchFraction = np.sum(agent.getAttrOfPeers('switch',LI_AA)) / N_FRIENDS
        inno, imit = agent.attr[['inno','imit']][0]
        
        if randNum < inno + ( imit * ( switchFraction)):
            agent.attr['switch'] = 1
            agent.attr['color']  = RED
            plotting.add(iStep,inno)
            
    if DO_PLOT and iStep%10 == 0:
        plotting.update(iStep, fracList, world.getAttrOfAgentType('color',agTypeID=AGENT))
    