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
N_AGENTS   = 500
N_STEPS    = 1000
MAX_EXTEND = 50

IMITATION = 15
INNOVATION = .2

N_FRIENDS  = 10

DEBUG = True
DO_PLOT  = True


BLUE = [0,0,1,1]
RED  = [1,0,0,1]

#%% setup
simNo, outputPath = core.setupSimulationEnvironment()

world = LIB.World(simNo,
              outputPath,
              spatial=True,
              nSteps=N_STEPS,
              maxNodes=1e4,
              maxLinks=1e5,
              debug=DEBUG)

AGENT = world.registerAgentType('agent' , AgentClass=LIB.Agent,
                               staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.float32, 2),
                                                    ('imit', np.float16, 1),
                                                    ('inno', np.float16,1)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
#%% Init of edge types
LI_AA = world.registerLinkType('ag-ag', AGENT,AGENT)

origins = [np.asarray(x) for x in [[25,25], [75,25], [25,75], [75,75]]]
individualDeltas  = np.clip(np.random.randn(2, N_AGENTS)*10, -25, 25)

for iAgent in range(N_AGENTS):
    
    sector = random.choice([0,1,2,3])
    
    origin = origins[sector]
    x,y = origin + individualDeltas[:,iAgent]
    inno = random.random() * INNOVATION
    imit = random.normalvariate(IMITATION,2 ) 
    
    agent = LIB.Agent(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE,
                      imit = imit,
                      inno = inno)
    
    agent.register(world)


#%% creation of spatial proximity network
  
positions = world.getAttrOfAgents('pos', agTypeID=AGENT)
agIDList  = world.getAgentIDs(AGENT)
innovationVal = world.getAttrOfAgents('inno', agTypeID=AGENT).astype(np.float64)

def network_creation(agent, world):
    
    # standard network inverse to spatial distance
    weig = np.sum((positions - agent.attr['pos'])**2,axis=1)
    weig = np.divide(1.,weig, out=np.zeros_like(weig), where=weig!=0)
    ##############################################
    # create a new creation rule for a network with the slowest possible diffusion
    
    
    #weig = 
    ##############################################
    
    
    # normalizing
    weig = weig / np.sum(weig)
    
    return weig
    
for agent in world.getAgents.byType(AGENT):
    weights = network_creation(agent, world)
    
    friendIDs = np.random.choice(agIDList, N_FRIENDS, replace=False, p=weights)
    
    [agent.addLink(ID, liTypeID = LI_AA) for ID in friendIDs]
    

    
    
positions = world.getAttrOfAgents('pos',agTypeID=AGENT)

##############################################
# exchange the position of spatial space (x,y) with the properties (inno, imit)

#positions[:,0] = 

##############################################

#%% Scheduler
iStep = 0
fracList = list()

if DO_PLOT:
    plotting = tools.PlotClass(positions, world, AGENT, LI_AA)
while True:
    tt =time.time()
    iStep+=1
    switched = world.getAttrOfAgents('switch',agTypeID=AGENT)
    switchFraction = np.sum(switched) / N_AGENTS
    fracList.append(switchFraction)
    
    
    if switchFraction == 1 or iStep == N_STEPS:
        break
    tools.printfractionExceed(switchFraction, iStep)
    
    nodesToIter = world.filterAgents(AGENT, 'switch', 'eq', 0)
    randValues  = np.random.random(len(nodesToIter))*1000
    
    for agent, randNum in zip(world.getAgents.byType(localIDs=nodesToIter),randValues) :
              
        switchFraction = np.sum(agent.getAttrOfPeers('switch',LI_AA)) / N_FRIENDS
        inno, imit = agent.attr[['inno','imit']][0]
        
        if randNum < inno + ( imit * ( switchFraction)):
            agent.attr['switch'] = 1
            agent.attr['color']  = RED
            plotting.add(iStep,inno)
            
    if DO_PLOT and iStep%50 == 0:
        plotting.update(iStep, fracList, world.getAttrOfAgents('color',agTypeID=AGENT))
    
    