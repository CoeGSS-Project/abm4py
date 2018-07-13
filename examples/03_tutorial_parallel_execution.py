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
sys.path.append('../')

from lib import World, Agent, Location, GhostLocation #, GhostAgent, World,  h5py, MPI
from lib.traits import Neighborhood, Collective, Mobile, Parallel
from lib import core

import tools_for_03 as tools

#%% SETUP
EXTEND = 50
RADIUS = 1.5

#%% Class definition
class Grass(Location, Collective, Parallel, Neighborhood):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Location.__init__(self, world, **kwAttr)
        Collective.__init__(self, world, **kwAttr)
        Parallel.__init__(self, world, **kwAttr)
        Neighborhood.__init__(self, world, **kwAttr)
    def add(self, value):
        
        self.attr['height'] += value


    def grow(self):
        currHeight = self.attr['height']
        for neigLoc in self.iterNeighborhood(ROOTS):
            if neigLoc['height'] > 2*currHeight:
                self['height'] *= ((random.random()*.8)+1)
                break
        else:
            self['height'] *= ((random.random()*.05)+1)

class GhostGrass(GhostLocation):   
    
    def __init__(self, world, **kwAttr):
        GhostLocation.__init__(self, world, **kwAttr)

    
#%% Init of world and register of agents and links
world = World(agentOutput=False,
              mpiComm=core.comm,
              maxNodes=100000,
              maxLinks=200000)

rankIDLayer = np.zeros([EXTEND, EXTEND]).astype(int)
if world.isParallel:
    print('parallel mode')
    rankIDLayer[EXTEND//2:,:EXTEND//2] = 1
    rankIDLayer[:EXTEND//2,EXTEND//2:] = 2
    rankIDLayer[:EXTEND//2,:EXTEND//2:] = 3
else:
    print('non-parallel mode')
    
world.setParameter('extend', EXTEND)
GRASS = world.registerAgentType('grass' , AgentClass=Grass, GhostAgentClass=GhostGrass,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('height', np.float32, 1)])
ROOTS = world.registerLinkType('roots',GRASS, GRASS, staticProperties=[('weig',np.float32,1)])

connBluePrint = world.spatial.computeConnectionList(radius=RADIUS)
world.spatial.initSpatialLayer(rankIDLayer, connBluePrint, Grass, ROOTS)

for grass in world.getAgents.byType(GRASS):
    grass.reComputeNeighborhood(ROOTS)
    if np.min(grass['pos']) < 6:
        grass['height'] = random.random()+ 3.1    
    else:
        grass['height'] = random.random()+ 0.1


plott = tools.PlotClass(world, rankIDLayer)
    
while True:
    tt = time.time()
    [grass.grow() for grass in world.getAgents.byType(GRASS)]
    world.papi.updateGhostAgents(propertyList=['height'])
    print(str(time.time() -tt) + ' s')
    


    plott.update(world)
