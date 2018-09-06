#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, Global Climate Forun e.V. (GCF)
http://www.globalclimateforum.org

This file is part of ABM4py.

ABM4py is free software: you can redistribute it and/or modify it 
under the terms of the GNU Lesser General Public License as published 
by the Free Software Foundation, version 3 only.

ABM4py is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>. 
GNU Lesser General Public License version 3 (see the file LICENSE).

@author: ageiges
"""

#%% load modules
import sys 
import os
import numpy as np
import time
import random

home = os.path.expanduser("~")

sys.path.append('../..')

#import the gcf abm library and core components
from abm4py import Agent, World # basic interface
from abm4py import core
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

#%% NEW AGENT CLASS DEFINTION
class Person(Agent):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Agent.__init__(self, world, **kwAttr)


    def createSocialNetwork(self, world):
        
        #opt1
        #distance = np.sum((positions - self.attr['pos'])**2,axis=1)
        
        #opt2
        distance = np.abs((innovationVal - self.attr['inno'])**4)
        
        # normalizing
        weights = np.divide(1.,distance, out=np.zeros_like(distance), where=distance!=0)
        weights = weights / np.sum(weights)
        
        
        friendIDs = np.random.choice(agIDList, N_FRIENDS, replace=False, p=weights)
        [self.addLink(ID, liTypeID = LI_AA) for ID in friendIDs]

#%% setup
simNo, outputPath = core.setupSimulationEnvironment()

# initialization of the world instance, with no 
world = World(simNo,
              outputPath,
              nSteps=N_STEPS,
              maxNodes=1e4,
              maxLinks=1e5,
              debug=DEBUG,
              agentOutput=True)

# register the first AGENT typ and save the numeric type ID as constant
AGENT = world.registerAgentType(AgentClass=Person,
                               staticProperties  = [('pos', np.float32, 2),
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
    agent = Person(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE,
                      imit = imit,
                      inno = inno)
    ##############################################
    
    # after the agent is created, it needs to register itself to the world
    # in order to get listed within the iterators and other predefined structures
    world.registerAgent(agent)

world.io.initAgentFile(world, [AGENT])
#%% creation of spatial proximity network

# world.getAttrOfAgentType is used to receive the position of all agents 
# for plotting. The label specifies the AGENT attribute and the agTypeID
# specifies the type of AGENT.
positions = world.getAttrOfAgentType('pos', agTypeID=AGENT)

# This produces a list of all agents by their IDs
agIDList  = world.getAgentIDs(AGENT)

# world.getAttrOfAgentType is used to receive the innovation value of all agents 
# for plotting. The label specifies the AGENT attribute and the agTypeID
# specifies the type of AGENT. The value is given as float.
innovationVal = world.getAttrOfAgentType('inno', agTypeID=AGENT).astype(np.float64)

# For a fixed agent this function assigns weights to all other agents 
# either by option 1 via proximity in position or by option 2 via proximity
# in innovation values.

 
# This loop then executes the choice of friends for every agent by picking 
# from the agent list according to the probability weights calculated by the 
# function above     
[agent.createSocialNetwork(world) for agent in world.getAgentsByType(AGENT)]
    

    
# Here one can choose the positions of the agents in the plot by putting all 
# options but the favoured one into comments.
    
positions = world.getAttrOfAgentType('pos',agTypeID=AGENT)
#positions[:,0] = world.getAttrOfAgentType('inno',agTypeID=AGENT)
#positions[:,1] = world.getAttrOfAgentType('imit',agTypeID=AGENT)
#%% Scheduler
iStep = 0
fracList = list()
fracPerSector = {1:[], 2:[], 3:[],4:[]}

switched = world.getAttrOfAgentType('switch',agTypeID=AGENT)


# If results shall be plotted:
if DO_PLOT:
    plotting = tools.PlotClass(positions, world, AGENT, LI_AA)
    
# The loop runs until one of the end-conditions below is fulfilled  
while True:
    tt =time.time()
    iStep+=1
    
    # world.getAttrOfAgentType is used to retrieve the attribute "switch" of all AGIDs
    switched = world.getAttrOfAgentType('switch',agTypeID=AGENT)
    
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
    #nodesToIter = world.filterAgents(AGENT, 'switch', 'eq', 0)
    agentsToIter = world.getAgentsByFilteredType(lambda a: a['switch'] == 0, AGENT, )
    randValues  = np.random.random(len(agentsToIter))*1000
    
    # instead of looping only over agents, we loop over packages of an agents
    # and it dedicated random number that the agent will use.    
    for agent, randNum in zip(agentsToIter,randValues) :
        
        # switchFraction is recalculated for every agent as the sum of all its
        # friends that switched devided by its total number of friends
        switchFraction = np.sum(agent.getAttrOfPeers('switch',LI_AA)) / N_FRIENDS
        inno = agent.attr['inno']
        imit = agent.attr['imit']
        
        # if the condition for an agent to switch is met, the agent attributes
        # "switch" and "color" are altered
        if randNum < inno + ( imit * ( switchFraction)):
            agent.attr['switch'] = 1
            agent.attr['color']  = RED
            plotting.add(iStep,inno)
    
    # each 50 steps, the visualization is updated       
    if DO_PLOT and iStep%50 == 0:
        plotting.update(iStep, fracList, world.getAttrOfAgentType('color',agTypeID=AGENT))
    
    world.io.writeAgentDataToFile(iStep, [AGENT])
    #time.sleep(.1)
    #print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))

world.io.finalizeAgentFile()
