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

from abm4py import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from abm4py.traits import Mobile
from abm4py import core
from abm4py import misc

BOLD = '\033[1m'
END = '\033[0m'
 #%% Setup
N_AGENTS   = 10000
REF_LENGTH = 1000
n_REPEAT   = 100
N_IDS      = 20
#%% register a new agent type with four attributes
world = World(agentOutput=False,
          maxNodes=100000,
          maxLinks=50000)

AGENT = world.registerAgentType(AgentClass=Agent,
                                 staticProperties  = [('float2', np.float64, 1),
                                                      ('float2', np.float64, 1)],
                                 dynamicProperties = [('integer', np.int64, 1),
                                                      ('integer_x2', np.int64, 2),
                                                      ('float', np.float64, 1),
                                                      ('float_x2', np.float64, 2),
                                                      ('float_x3', np.float64, 3),
                                                      ('float_x4', np.float64, 4)])

for iAgent in range(N_AGENTS):
    
    agent = Agent(world)
    agent.register(world)


print(' ######################## TESTING OF AGENT TYPE ###############################')

print('Reading write of attributes of type integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    x = world.getAttrOfAgentType('integer', AGENT)
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Reading attributes of type float', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    x = world.getAttrOfAgentType('float', AGENT)
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Writing attributes of type integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    world.setAttrOfAgentType('integer', x+2, AGENT)
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Writing attributes of type float', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    world.setAttrOfAgentType('float', x+1, AGENT)
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print(' ######################## TESTING LIST OF IDS ###############################')

#%%
print('Reading attributes of type integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    agentIDs = np.random.choice(world.getAgentIDs(AGENT),N_IDS)
    tt = time.time()
    x = world._graph.getAttrOfNodesSeq('integer', agentIDs)
    timeReq.append(time.time() -tt)
print('Average: {:3.6f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))


print('Reading attributes of type integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    agentIDs = np.random.choice(world.getAgentIDs(AGENT),N_IDS)
    tt = time.time()
    y = world._graph.getNodeSeqAttr('integer', agentIDs)
    timeReq.append(time.time() -tt)
print('Average: {:3.6f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))


   
print('Writing attributes of type integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    agentIDs = np.random.choice(world.getAgentIDs(AGENT),N_IDS)
    tt = time.time()
    world._graph.setAttrOfNodesSeq('integer', x+4, agentIDs)
    timeReq.append(time.time() -tt)
print('Average: {:3.6f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

   
print('Writing attributes of type integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    agentIDs = np.random.choice(world.getAgentIDs(AGENT),N_IDS)
    tt = time.time()
    world._graph.setNodeSeqAttr('integer', x+4, agentIDs)
    timeReq.append(time.time() -tt)
print('Average: {:3.6f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))


