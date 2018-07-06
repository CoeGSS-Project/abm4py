#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:27:20 2018

@author: geiges
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

from lib import World, Agent
import tools_for_01 as tools

#%%
world = World(agentOutput=False)
#%% register a new agent type with four attributes
AGENT_ID = world.registerNodeType('agent' , AgentClass=Agent,
                               staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.int16, 2)],
                               dynamicProperties = [('age', np.int16, 1),
                                                    ('name', np.str,10)])
#%% register a link type to connect agents
LINK = world.registerLinkType('link',AGENT_ID, AGENT_ID)
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
tools.plotGraph(world, AGENT_ID, LINK, attrLabel= 'age')

# make new connection from clara to joana
clara.addLink(joana.nID,LINK)
#%%
tools.plotGraph(world, AGENT_ID, LINK, attrLabel= 'age')

clara.getPeerAttr('age', LINK)

#%%
#getting ages of everbody
ages = world.getNodeAttr('age',nodeTypeID=AGENT_ID)

# get all canditates
candidates = world.getNode(nodeTypeID=AGENT_ID) 
ages = world.getNodeAttr('age', nodeTypeID=AGENT_ID)

# compute probability to connect for all other agents
differenceInAge = np.abs(ages -joana.get('age'))
probabilityToConnect = 1 / differenceInAge 
probabilityToConnect[np.isinf(probabilityToConnect)] = 0
probabilityToConnect = probabilityToConnect / np.sum(probabilityToConnect)
newIDToLink = np.random.choice(candidates, p=probabilityToConnect)

# connect new agent
joana.addLink(newIDToLink, LINK)
tools.plotGraph(world, AGENT_ID, LINK, attrLabel= 'age')
        