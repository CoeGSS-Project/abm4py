#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2017
Global Climate Forum e.V.
http://www.globalclimateforum.org

This file is part of GCFABM.

GCFABM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GCFABM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCFABM.  If not, see <http://www.gnu.org/licenses/>.

"""


#%% load modules

import sys 
import os
import numpy as np
import logging as lg
import time
import random
import h5py
from math import sqrt
import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../../')


from lib import World, Agent, Location #, GhostAgent, World,  h5py, MPI
from lib.traits import Aggregator
from lib import core

#import tools_for_02 as tools

#%% SETUP
EXTEND = 30
GRASS_PER_PATCH = 30
N_REPEAT = 30
#%% classes
#Patch = Location
class Patch(Aggregator, Location):
    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
       
        Location.__init__(self, world, **kwAttr) 
        Aggregator.__init__(self, world, **kwAttr) 
    
class Grass(Agent):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Location.__init__(self, world, **kwAttr) 

        
    def add(self, value):
        
        self.attr['height'] += value


    def grow(self):
        """        
        The function grow lets the grass grow by ten percent.
        If the grass height is smaller than 0.1, and a neighboring patch has 
        grass higher than 0.7, the grass grows by .05. Then it grows by 
        10 percent.
        """
                
        if self.attr['height'] < 0.1:
            for neigLoc in self.iterNeighborhood(ROOTS):
                if neigLoc.attr['height'] > 0.9:
                    self['height'] += 0.05
                    
                    if self['height'] > 0.1:
                        break
                    
        self['height'] = min(self['height']*1.1, 1.)
#%%
world = World(agentOutput=False,
                  maxNodes=100000,
                  maxLinks=1000000)

world.setParameter('extend', EXTEND)
#%% register a new agent type with four attributes
PATCH = world.registerAgentType('PATCH' , AgentClass=Patch,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('sumGrass', np.float64, 1)])


GRASS = world.registerAgentType('flower' , AgentClass=Grass,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('height', np.float64, 1)])
#%% register a link type to connect agents

PATCHWORK = world.registerLinkType('patchwork',PATCH, PATCH, staticProperties=[('weig',np.float32,1)])

ROOTS     = world.registerLinkType('roots',PATCH, GRASS)
IDArray = np.zeros([EXTEND, EXTEND])



tt = time.time()
for x in range(EXTEND):
    for y in range(EXTEND):
        
        patch = Patch(world, 
                      pos=(x,y),
                      sumGrass=0)
        patch.register(world)
        world.registerLocation(patch, x,y)
        IDArray[x,y] = patch.nID
        
        for i in range(GRASS_PER_PATCH):
            grass = Grass(world,
                          pos= (x,y),
                          height = random.random())
            grass.register(world)
            patch.addLink(grass.nID, ROOTS)
        
connBluePrint = world.spatial.computeConnectionList(radius=4.5)
world.spatial.connectLocations(IDArray, connBluePrint, PATCHWORK, Patch)
print('init Patches in: ' + str(time.time() - tt))


#%%
tt = time.time()
for i in range(N_REPEAT):
    x = list()
    for patch in world.getAgents.byType(PATCH):
        x.append(np.mean(patch.getAttrOfPeers('height', ROOTS)))
timeWithoutAggr = time.time() - tt 
print(timeWithoutAggr)


tt = time.time()
for i in range(N_REPEAT):
    x2 = list()
    
    for patch in world.getAgents.byType(PATCH):
        x2.append(np.mean([item['height'] for item in patch.aggegationDict[ROOTS]]))
timeWithAggr = time.time() - tt 

print(timeWithAggr)
print('Factor: ' + str(timeWithAggr / timeWithoutAggr ))


