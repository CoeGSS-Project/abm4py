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
import random

from gcfabm import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from gcfabm.traits import Mobile
from gcfabm import core

 #%% Setup
N_AGENTS   = 50000
REF_LENGTH = 1000
n_REPEAT   = 50
#%% register a new agent type with four attributes
world = World(agentOutput=False,
          maxNodes=100000,
          maxLinks=500000)

AGENT = world.registerAgentType(AgentClass=Agent,
                               staticProperties  = [('randNum')],
                               dynamicProperties = [])

for iAgent in range(N_AGENTS):
    
    agent = Agent(world,
                  randNum = random.random())
    agent.register(world)
    
tt = time.time()   
resultNew = dict() 
for treshold in np.linspace(0,1):
    resultNew[treshold] = len(world.filterAgents(AGENT, lambda a: a['randNum'] < treshold))
print('Agent filered 50 times in  ' + str(time.time() -tt) )      

resultOld = dict()
tt = time.time()    
for treshold in np.linspace(0,1):
    resultOld[treshold] = len(world.getAgents.byFilter(AGENT, 'randNum', 'lt', treshold))
    #print(x)
print('OLD - Agent filered 50 times in  ' + str(time.time() -tt)) 

for treshold in np.linspace(0,1):
    assert resultOld[treshold] == resultNew[treshold]