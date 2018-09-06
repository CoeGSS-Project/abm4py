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

import numpy as np
import time

from gcfabm import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from gcfabm.traits import Mobile
from gcfabm import core

 #%% Setup
N_AGENTS   = 10000
REF_LENGTH = 1000
n_REPEAT   = 50
#%% register a new agent type with four attributes
world = World(agentOutput=False,
          maxNodes=100000,
          maxLinks=500000)

AGENT = world.registerAgentType(AgentClass=Agent,
                                         staticProperties  = [],
                                         dynamicProperties = [])

for iAgent in range(N_AGENTS):
    
    agent = Agent(world)
    agent.register(world)
    
print('Asking reference for a list of IDs')    
idList = world.getAgentIDs(agTypeID=AGENT)
timeReq = list()   
maxNodes = world.maxNodes
for iTry in range(30):
    tt = time.time()
    for i in range(1000):
        subList = idList[i:i+REF_LENGTH]
        x = world.graph.getNodeDataRef(subList)
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Asking reference for a array of IDs')    
timeReq = list()  
idList = np.asarray(world.getAgentIDs(agTypeID=AGENT))
for iTry in range(30):
    tt = time.time()
    for i in range(1000):
        subList = idList[i:i+REF_LENGTH]
        x = world.graph.getNodeDataRef(subList)
        
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))


print('Asking reference for a single ID')    
timeReq = list()  
idList =world.getAgentIDs(agTypeID=AGENT)
for iTry in range(30):
    tt = time.time()
    for i in range(1000):
        singleID = idList[i]
        x = world.graph.getNodeDataRef(singleID)
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
