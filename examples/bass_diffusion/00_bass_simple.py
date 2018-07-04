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

#%% import of modules
import sys 
import os
import numpy as np
import time

home = os.path.expanduser("~")
sys.path.append('../..')

#import the gcf abm library and core components
import lib as lib # basic interface
import tools

#%% CONFIG
N_AGENTS   = 500 # number of AGID that will be gernerated
N_STEPS    = 1000 # number of steps performed
MAX_EXTEND = 50  # spatial extend 

IMITATION = 10
INNOVATION = .1

BLUE = [0,0,1,1]
RED  = [1,0,0,1]

#%% setup

# initialization of the world instance, with no 
world = lib.World(wAgentOutput=False)

# register the first AGID typ and save the numeric type ID as constant
AGID = world.registerNodeType('Agent' , AgentClass=lib.Agent,
                               staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.int16, 2)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])

#%% AGID creation

# looping over the number of AGIDs set up
for iAgent in range(N_AGENTS):
    
    # randomly draw and x,y position within the defined spatial extend
    x,y = np.random.randint(0, MAX_EXTEND, 2)

    ##############################################
    #create all AGID with tree properties
    # - pos = x,y
    # - switch 
    # - color = BLUE

    # LIB.AGID is the basic predefined AGID class. More complex classes can
    # be inherted from that one.
    # The init of LIB.AGIDs requires either the definition of all attributes 
    # that are registered (above) or none.
    agent = lib.Agent(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE)
    ##############################################
    
    # after the AGIDs is created, it needs to register itself to the world
    # in order to get listed within the iterators and other predefined structures
    agent.register(world)
    

#%% Scheduler
fracList = list()

# world.getNodeAttr is used to receive the position of all agents 
# for plotting. The label specifies the AGID attribute and the nodeTypeID
# specifies the type of AGID.
positions = world.getNodeAttr(label='pos',nodeTypeID=AGID)

# this class is only implemented for a convenient interactive visualization of 
# the example
ploting = tools.PlotClass(positions, world, AGID)

tt =time.time()

# this loop executes the specified number of steps 
for iStep in range(N_STEPS):
    
    
    # world.getNodeAttr is used to retrieve the attribute "switch"  of all AGIDs
    switched = world.getNodeAttr('switch',nodeTypeID=AGID)
    
    # the sum of all agents that switched, devided by the total number of agents
    # calculates the fraction of agents that already switched
    switchFraction = np.sum(switched) / N_AGENTS
    # the fraction is appended to the list for recording and visualization
    fracList.append(float(switchFraction))
    
    # this implements an additional end-condition to avoid running the model
    # without any active agents
    if switchFraction == 1:
        break
    
    # for a bit of speed up, we draw the required random numbers before 
    # the actual loop over agents.
    randomNumbers = np.random.random(N_AGENTS)*1000
    
    # instead of looping only over agents, we loop over packages of an agents
    # and it dedicated random number that the agent will use.
    for agent, randNum in zip(world.iterNodes(AGID), randomNumbers):
        
        # if the agent did not switch yet, we compute the new probability
        # to swich this step
        if agent.attr['switch'] == 0:
            
            # implemnent the probability of AGIDs to switch (actually multiplied 
            # by a factor of 1000)
            probability = INNOVATION + ( IMITATION * (switchFraction ))
            
            # if the condition is met, the agent attributes "switch" and "color"
            # are altered
            if randNum < probability:
                agent.attr['switch'] = 1
                agent.attr['color'] = RED
    
    # each 50 steos, the visualization is updated        
    if iStep%50 == 0:
        ploting.update(iStep+1, fracList, world.getNodeAttr('color',nodeTypeID=AGID))
    
    print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))