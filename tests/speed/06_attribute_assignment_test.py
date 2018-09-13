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

import sys 
import numpy as np
import time

from abm4py import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from abm4py.traits import Mobile
from abm4py import core

BOLD = '\033[1m'
END = '\033[0m'
 #%% Setup
N_AGENTS   = 50000
REF_LENGTH = 1000
n_REPEAT   = 30
#%% register a new agent type with four attributes
world = World(agentOutput=False,
          maxNodes=100000,
          maxLinks=50000)

AGENT = world.registerAgentType(AgentClass=Agent,
                                         staticProperties  = [],
                                         dynamicProperties = [('integer', np.int64, 1),
                                                              ('integer_x2', np.int64, 2),
                                                              ('float', np.float, 1),
                                                              ('float_x2', np.float, 2)])

for iAgent in range(N_AGENTS):
    
    agent = Agent(world)
    agent.register(world)



#%% METHOD: agent.attr[]
print('################### Assignment method: "agent.attr[]" ###################')        
print('Assigning single integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    for i, agent in enumerate(world.getAgentsByType(AGENT)):
        agent.attr['integer'] = i+iTry
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Assigning double integer', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    for i, agent in enumerate(world.getAgentsByType(AGENT)):
        agent.attr['integer_x2'] = i,i+iTry
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Assigning single float', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    for i, agent in enumerate(world.getAgentsByType(AGENT)):
        agent.attr['float'] = i+iTry
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Assigning double float', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    for i, agent in enumerate(world.getAgentsByType(AGENT)):
        agent.attr['float_x2'] = i,i+iTry
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

print('Assigning float and interger seperately', end=' ')    
timeReq = list()
for iTry in range(n_REPEAT):
    tt = time.time()
    for i, agent in enumerate(world.getAgentsByType(AGENT)):
        agent.attr['float'] = iTry
        agent.attr['integer'] = i+iTry
    timeReq.append(time.time() -tt)
print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

#print('Assigning float and interger jointly', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.attr[['float', 'integer']] = iTry, i+iTry
#        
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

#%% METHOD: agent[]
#print('################### Assignment method: "agent[]" ###################')        
#print('Assigning single integer', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent['integer'] = i+iTry
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning double integer', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent['integer_x2'] = i,i+iTry
#    timeReq.append(time.time() -tt)
#print('Average: '+ BOLD + ' {:3.4f}'.format(np.mean(timeReq[1:])) + END +' s STD: {:3.4f} s'.format(np.std(timeReq[1:])))
#
#print('Assigning single float', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent['float'] = i+iTry
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning double float', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent['float_x2'] = i,i+iTry
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning float and interger seperately', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent['float'] = iTry
#        agent['integer'] = i+iTry
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

#print('Assigning float and interger jointly', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent[['float', 'integer']] = iTry, i+iTry
#        
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

#%% METHOD: agent.set
#print('################### Assignment method: "agent.set()" ###################')        
#print('Assigning single integer', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.set('integer', i+iTry)
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning double integer', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.set('integer_x2', (i,i+iTry))
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning single float', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.set('float', i+iTry)
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning double float', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.set('float_x2', (i,i+iTry))
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))
#
#print('Assigning float and interger seperately', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.set('float', iTry)
#        agent.set('integer', i+iTry)
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

#print('Assigning float and interger jointly', end=' ')    
#timeReq = list()
#for iTry in range(n_REPEAT):
#    tt = time.time()
#    for i, agent in enumerate(world.getAgentsByType(AGENT)):
#        agent.set(['float', 'integer'], (iTry, i+iTry))
#        
#    timeReq.append(time.time() -tt)
#print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))