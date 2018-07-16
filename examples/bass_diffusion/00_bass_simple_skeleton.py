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

import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../../lib/')

import lib_gcfabm_prod as LIB #, GhostAgent, World,  h5py, MPI
import core_prod as core
import tools

#%% CONFIG
N_AGENTS   = 500
N_STEPS    = 1000
MAX_EXTEND = 50

IMITATION = 10
INNOVATION = .1

DEBUG = True

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
                                                    ('pos', np.int16, 2)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
# %%

for iAgent in range(N_AGENTS):
    
    x,y = np.random.randint(0, MAX_EXTEND, 2)
    
    ##############################################
    #create all agent with tree properties
    # - pos = x,y
    # - switch 
    # - color = BLUE
    

    agent =
    
    ##############################################
    agent.register(world)
    

#%% Scheduler
iStep = 0
fracList = list()

##############################################
# get position of all agents for plotting

positions = 

##############################################


ploting = tools.PlotClass(positions, world,AGENT)

while True:
    tt =time.time()
    iStep+=1
    
    
    ##############################################
    #calculate the fraction of agents that already switched
    
    switchFraction = 
    
    ##############################################
    
    fracList.append(switchFraction)
    
    if switchFraction == 1 or iStep == N_STEPS:
        break
    
    
    for agent, randNum in zip(world.getAgents.byType(AGENT), np.random.random(N_AGENTS)*1000):
        
        if agent.attr['switch'] == 0:
            
            ##############################################
            # implemnent the condition of agents to switch
            
            condition = 
            
            ##############################################
            
            if randNum < condition:
                agent.attr['switch'] = 1
                agent.attr['color'] = RED
            
    if iStep%50 == 0:
        ploting.update(iStep, fracList, world.getAttrOfAgentType('color',agTypeID=AGENT))
    
    print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))