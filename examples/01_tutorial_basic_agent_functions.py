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
import logging as lg
import time
import random
import h5py

import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../')

from abm4py import World, Agent
import tools_for_01 as tools

#%%
world = World(agentOutput=False)
#%% register a new agent type with four attributes
HUMANS = world.registerAgentType(Agent, 
                                 agTypeStr = 'human',
                                 staticProperties  = [('pos', np.int16, 2)],
                                 dynamicProperties = [('age', np.int16, 1),
                                                      ('name', np.str,10)])
#%% register a link type to connect agents
LINK = world.registerLinkType('link',HUMANS, HUMANS)
print(LINK)

#%% create an agent called theodor

theodor = Agent(world,
                    pos=np.random.randint(0, 50, 2),
                    age = 25,
                    name = 'theodor')    
    
theodor.register(world)
#%% create an agent called clara

clara = Agent(world,    
                  pos=  np.random.randint(0, 50, 2),
                  age = 33,
                  name = 'clara')    
    
clara.register(world)
#%%create an agent called joana
joana = Agent(world,    
                  pos=np.random.randint(0, 50, 2),
                  age = 27 ,
                  name = 'joana')    
    
joana.register(world)

#%%
tools.plotGraph(world, HUMANS, LINK, attrLabel= 'age')

# make new connection from clara to joana
clara.addLink(joana.ID,LINK)
#%%
tools.plotGraph(world, HUMANS, LINK, attrLabel= 'age')

print('Claras peers have the age of: ' + str(clara.getAttrOfPeers('age', LINK)))

#%%
#getting ages of everbody
ages = world.getAttrOfAgentType('age',agTypeID=HUMANS)

# get all canditates

# get all canditates:
# stf: ages ist ein np.array, aber die IDs sind eine python Liste :-(
# candidateIDs != joana.ID ergibt dann nämlich True, anstatt [True, True, False]
# stf: ausserdem existiert zwar gID in dem np array, aber ist nicht gesetzt 
candidateIDs = world.getAgentIDs(HUMANS)

candidateIsJoana = [id == joana.ID for id in candidateIDs]

# compute probability to connect for all other agents
differenceInAge = np.abs(ages - joana.attr['age'])
probabilityToConnect = 1 / (differenceInAge + 0.01) 
probabilityToConnect[candidateIsJoana] = 0
probabilityToConnect = probabilityToConnect / np.sum(probabilityToConnect)
newIDToLink = np.random.choice(candidateIDs, p = probabilityToConnect)

# connect new agent
joana.addLink(newIDToLink, LINK)
tools.plotGraph(world, HUMANS, LINK, attrLabel= 'age')
