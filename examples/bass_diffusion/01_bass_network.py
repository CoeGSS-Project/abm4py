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

import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../../lib/')

import lib_gcfabm_prod as LIB #, Agent, World,  h5py, MPI
import core_prod as core
import tools
#%% CONFIG
N_AGENTS   = 500    # number of AGID that will be gernerated
N_STEPS    = 1000   # number of steps performed
MAX_EXTEND = 50     # spatial extend 

IMITATION = 15
INNOVATION = .2

N_FRIENDS  = 10     # number of friends/connections an agent will have

DEBUG = True
DO_PLOT  = True     # decides whether to lateron plot the results or not


BLUE = [0,0,1,1]
RED  = [1,0,0,1]

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

# register the first AGENT typ and save the numeric type ID as constant
AGENT = world.registerNodeType('agent' , AgentClass=LIB.Agent,
                               staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.float32, 2),
                                                    ('imit', np.float16, 1),
                                                    ('inno', np.float16,1)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
#%% Init of edge types
LI_AA = world.registerLinkType('ag-ag', AGENT,AGENT)
#%% Agent creation

# set position of the agents into four sector centres:
origins = [np.asarray(x) for x in [[25,25], [75,25], [25,75], [75,75]]]
# normal distribution around centres:
individualDeltas  = np.clip(np.random.randn(2, N_AGENTS)*10, -25, 25)


# looping over the number of AGENTs set up
for iAgent in range(N_AGENTS): 
    
    # agent chooses one of the sectors in the spatial extend    
    sector = random.choice([0,1,2,3])
    
    # agent is assigned a position around the centre of the choosen sector
    origin = origins[sector]
    x,y = origin + individualDeltas[:,iAgent]
    
    # agent is assigned personal innovation and imitation levels
    inno = random.random() * INNOVATION
    imit = random.normalvariate(IMITATION,2 )
    
    ##############################################
    #create all agents with properties
    # - pos = x,y
    # - switch 
    # - color = BLUE
    # - imit 
    # - inno

    # The init of LIB.Agent requires either the definition of all attributes 
    # that are registered (above) or none.
    agent = LIB.Agent(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE,
                      imit = imit,
                      inno = inno)
    ##############################################
    
    # after the agent is created, it needs to register itself to the world
    # in order to get listed within the iterators and other predefined structures
    agent.register(world)


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

# For a fixed agent this function assigns weights to all other agents 
# either by option 1 via proximity in position or by option 2 via proximity
# in innovation values.
def network_creation(agent, world):
    
    #opt1
    #weig = np.sum((positions - agent.attr['pos'])**2,axis=1)
    
    #opt2
    weig = np.abs((innovationVal - agent.attr['inno'])**4)
    
    # normalizing
    weig = np.divide(1.,weig, out=np.zeros_like(weig), where=weig!=0)
    weig = weig / np.sum(weig)
    
    return weig
 
# This loop then executes the choice of friends for every agent by picking 
# from the agent list according to the probability weights calculated by the 
# function above    
for agent in world.iterNodes(AGENT):
    
    weights = network_creation(agent, world)
    
    friendIDs = np.random.choice(agIDList, N_FRIENDS, replace=False, p=weights)
    
    [agent.addLink(ID, linkTypeID = LI_AA) for ID in friendIDs]
    

    
# Here one can choose the positions of the agents in the plot by putting all 
# options but the favoured one into comments.
    
positions = world.getNodeAttr('pos',nodeTypeID=AGENT)
#positions[:,0] = world.getNodeAttr('inno',nodeTypeID=AGENT)
#positions[:,1] = world.getNodeAttr('imit',nodeTypeID=AGENT)
#%% Scheduler
iStep = 0
fracList = list()
fracPerSector = {1:[], 2:[], 3:[],4:[]}

switched = world.getNodeAttr('switch',nodeTypeID=AGENT)


# If results shall be plotted:
if DO_PLOT:
    plotting = tools.PlotClass(positions, world, AGENT, LI_AA)
    
# The loop runs until one of the end-conditions below is fulfilled  
while True:
    tt =time.time()
    iStep+=1
    
    # world.getNodeAttr is used to retrieve the attribute "switch" of all AGIDs
    switched = world.getNodeAttr('switch',nodeTypeID=AGENT)
    
    # the sum of all agents that switched, devided by the total number of agents
    # calculates the fraction of agents that already switched
    switchFraction = np.sum(switched) / N_AGENTS
    
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
        
        # switchFraction is recalculated for every agent as the sum of all its
        # friends that switched devided by its total number of friends
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
    
    #time.sleep(.1)
    #print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))