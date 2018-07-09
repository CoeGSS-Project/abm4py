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
import os
import numpy as np
import time
import random
import h5py

import matplotlib.pyplot as plt
home = os.path.expanduser("~")

#import the gcf abm library and core components
import lib as LIB # basic interface
from lib import core
import tools

#%% CONFIG
N_AGENTS   = 1000   # number of AGID that will be gernerated
N_STEPS    = 1000   # number of steps performed
MAX_EXTEND = 50     # spatial extend 

IMITATION = 15
INNOVATION = .05

N_FRIENDS  = 10     # number of friends/connections an agent will have

REDUCTION_FACTOR = 100000

DEBUG = True
DO_PLOT  = True     # decides whether to lateron plot the results or not


BLUE = plt.get_cmap('Blues')(.3)
RED  = plt.get_cmap('RdPu_r')(.1)

#%% setup
simNo, outputPath = core.setupSimulationEnvironment()

# initialization of the world instance, with no 
world = LIB.World(simNo,
              outputPath,
              spatial=True,
              nSteps=N_STEPS,
              maxNodes=1e4,
              maxLinks=1e5,
              debug=DEBUG)

# register the first AGENT type and save the numeric type ID as constant further referred to as nodeTypeID
CELL = world.registerNodeType('cell' , AgentClass=LIB.Location,
                              staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.float32, 2),
                                                    ('imit', np.float16, 1),
                                                    ('nAgents', np.int16,1)],
                              dynamicProperties = [('fraction', np.int16, 1)])

# register the first AGENT typ and save the numeric type ID as constant
AGENT = world.registerNodeType('agent' , AgentClass=LIB.Agent,
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
connBluePrint = world.spatial.computeConnectionList(radius=1)
world.spatial.connectLocations(IDArray, connBluePrint, LI_CC, CELL)

if True:
    #%%
    tools.plotGraph(world, CELL, LI_CC, 'nAgents')
#%% Agent creation

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
# change the propertyToPreference function so 
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
        age    = personData[currIdx, H5_AGE]
        gender = personData[currIdx, H5_GENDER]
        income  = personData[currIdx, H5_INCOME]
        hhType  = personData[currIdx, H5_HHTYPE]
    
        inno, imit = propertyToPreference(age, gender, income, hhType)
        
        ##############################################
        #create all agents with properties
        # - pos = x,y
        # - switch 
        # - color = BLUE
        # - imit 
        # - inno
        # - nPers
        # - age
        # - income
        # - gender

        # The init of LIB.Agent requires either the definition of all attributes 
        # that are registered (above) or none.
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
        ##############################################
    
        # after the agent is created, it needs to register itself to the world
        # in order to get listed within the iterators and other predefined structures
        agent.register(world)
        currIdx +=1

#%% creation of spatial proximity network

# world.getNodeAttr is used to receive the position of all agents 
# for plotting. The label specifies the AGENT attribute and the nodeTypeID
# specifies the type of AGENT.  
positions = world.getNodeAttr('pos', nodeTypeID=AGENT)

# This produces a list of all agents by their IDs
agIDList  = world.getNodeIDs(AGENT)

# world.getNodeAttr is used to receive the innovation value of all agents 
# for plotting. The label specifies the AGENT attribute and the nodeTypeID
# specifies the type of AGENT. The value is given as float.
innovationVal = world.getNodeAttr('inno', nodeTypeID=AGENT).astype(np.float64)

for agent in world.iterNodes(AGENT):
    ##############################################
    # create a new creation rule 
    
    # spatial weight
    weig1 = np.sum((positions - agent.attr['pos'])**2,axis=1)
    weig1 = np.divide(1.,weig1, out=np.zeros_like(weig1), where=weig1!=0)
    
    # preference weight
    weig2 = np.abs((innovationVal - agent.attr['inno'])**2)
    weig2 = np.divide(1.,weig2, out=np.zeros_like(weig2), where=weig2!=0)
    
    # merging of weights
    weig = weig1 * weig2
    weig = weig / np.sum(weig)

    ##############################################
    
    
    friendIDs = np.random.choice(agIDList, N_FRIENDS, replace=False, p=weig)
    
    [agent.addLink(ID, linkTypeID = LI_AA) for ID in friendIDs]
    

        
    
positions = world.getNodeAttr('pos',nodeTypeID=AGENT)

##############################################
# exchange the position of spatial space (x,y) with the properties (inno, imit)

#positions[:,0] = 

##############################################


if False:
    #%%
    plt.figure('statistics')
    plt.subplot(2,2,1)
    data = world.getNodeAttr('age',nodeTypeID=AGENT)
    plt.hist(data)
    plt.title('age distribution')
    plt.subplot(2,2,2)
    data = world.getNodeAttr('income',nodeTypeID=AGENT)
    plt.hist(data)
    plt.title('income distribution')
    plt.subplot(2,2,3)
    data = world.getNodeAttr('nPers',nodeTypeID=AGENT)
    plt.hist(data)
    plt.title('household size')
    plt.subplot(2,2,4)
    data = world.getNodeAttr('nPers',nodeTypeID=AGENT)
    plt.scatter(world.getNodeAttr('income',nodeTypeID=AGENT), world.getNodeAttr('age',nodeTypeID=AGENT))
    
    plt.title('relation income to age')
    plt.draw()

#%% Scheduler
iStep = 0
fracList = list()

# If results shall be plotted:
if DO_PLOT:
    plotting = tools.PlotClass(positions, world, AGENT, LI_AA)

yData = [0.0233236, 0.0291545, 0.0612245 ,0.116618 ,0.157434 ,0.204082 ,0.282799,0.379009,
0.489796 ,0.600583 ,0.64431 ,0.644315,0.6559]
xData = [6*x for x in range(12,len(yData)*12+1,12)]

plotting.add_data(xData,yData)

# The loop runs until one of the end-conditions below is fulfilled
while True:
    tt =time.time()
    iStep+=1
    
    # world.getNodeAttr is used to retrieve the attribute "switch" of all AGIDs
    switched = world.getNodeAttr('switch',nodeTypeID=AGENT)
    
    # the sum of all agents that switched, devided by the total number of agents
    # calculates the fraction of agents that already switched
    switchFraction = np.sum(switched) / world.nNodes(AGENT)
    
    # the fraction is appended to the list for recording and visualization
    fracList.append(switchFraction)
    
    # this implements the end-conditions for the loop. It stops after the 
    # given amount of steps and also avoids running the model without any 
    # active agents
    if switchFraction == 1 or iStep == N_STEPS:
        break
    tools.printfractionExceed(switchFraction, iStep)
    
    # for a bit of speed up, we draw the required random numbers before 
    # the actual loop over agents.
    nodesToIter = world.filterNodes(AGENT, 'switch', 'eq', 0)
    randValues  = np.random.random(len(nodesToIter))*1000
    
    # instead of looping only over agents, we loop over packages of an agents
    # and it dedicated random number that the agent will use.  
    for agent, randNum in zip(world.iterNodes(localIDs=nodesToIter),randValues) :
        
        # dynamic of the agent
        switchFraction = np.sum(agent.getPeerAttr('switch',LI_AA)) / N_FRIENDS
        inno, imit = agent.attr[['inno','imit']][0]
        
        # if the condition for an agent to switch is met, the agent attributes
        # "switch" and "color" are altered
        if randNum < inno + ( imit * ( switchFraction)):
            agent.attr['switch'] = 1
            agent.attr['color']  = RED
            plotting.add(iStep,inno)
    
    # each 50 steps, the visualization is updated               
    if DO_PLOT and iStep%50 == 0:
        plotting.update(iStep, fracList, world.getNodeAttr('color',nodeTypeID=AGENT))
    