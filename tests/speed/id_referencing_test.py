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

import sys 
import numpy as np
import time
import random
sys.path.append('../../')

from lib import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from lib.enhancements import Mobile
from lib import core

 #%% Setup
N_AGENTS = 10000
REF_LENGTH = 1000
#%% register a new agent type with four attributes
world = World(agentOutput=False,
          maxNodes=100000,
          maxLinks=500000)

AGENT = world.registerAgentType('agent' , AgentClass=Agent,
                               staticProperties  = [],
                               dynamicProperties = [])

for iAgent in range(N_AGENTS):
    
    agent = Agent(world)
    agent.register(world)
    
print('list of IDs')    
idList = world.getAgentIDs(agTypeID=AGENT)
timeReq = list()   
maxNodes = world.maxNodes
for iTry in range(30):
    tt = time.time()
    for i in range(1000):
        subList = idList[i:i+REF_LENGTH]
        x = world.graph.getEdgeDataRef(subList)
    timeReq.append(time.time() -tt)
print(str(np.mean(timeReq)) + ' +- '  + str(np.std(timeReq)))

print('array of IDs')    
idList = np.asarray(world.getAgentIDs(agTypeID=AGENT))
for iTry in range(30):
    tt = time.time()
    for i in range(1000):
        subList = idList[i:i+REF_LENGTH]
        x = world.graph.getEdgeDataRef(subList)
    timeReq.append(time.time() -tt)
print(str(np.mean(timeReq)) + ' +- '  + str(np.std(timeReq)))

print('single ID')    
idList =world.getAgentIDs(agTypeID=AGENT)
for iTry in range(30):
    tt = time.time()
    for i in range(1000):
        singleID = idList[i]
        x = world.graph.getEdgeDataRef(singleID)
    timeReq.append(time.time() -tt)
print(str(np.mean(timeReq)) + ' +- '  + str(np.std(timeReq)))