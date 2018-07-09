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
from lib.enhancements import Neighborhood, Collective, Mobile, Parallel
from lib import core

import tools_for_03 as tools

#%% SETUP
EXTEND = 100


#%% Class definition
class Grass(Location, Collective, Parallel):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Location.__init__(self, world, **kwAttr)
        Collective.__init__(self, world, **kwAttr)
        Parallel.__init__(self, world, **kwAttr)
        
    def add(self, value):
        
        self.attr['height'] += value


    def grow(self):
        """ Jette
        
        The function grow lets the grass grow by ten percent.
        If the grass height is higher than 0.7 it lets random neighbours
        grow by the length 0.1
        
        """
        

        currHeight = self.attr['height']
        for neigLoc in self.iterNeighborhood(ROOTS):
            if neigLoc.attr['height'] > 2*currHeight:
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
                               staticProperties  = [('gID', np.int32, 1),
                                                    ('pos', np.int16, 2),
                                                    ('instance', np.object_,1)],
                               dynamicProperties = [('height', np.float32, 1)])
ROOTS = world.registerLinkType('roots',GRASS, GRASS, staticProperties=[('weig',np.float32,1)])

connBluePrint = world.spatial.computeConnectionList(radius=1.5)
world.spatial.initSpatialLayer(rankIDLayer, connBluePrint, Grass, ROOTS)
#plt.pcolormesh(rankIDLayer)
print(world.nAgents())
for grass in world.iterNodes(GRASS):
    grass.reComputeNeighborhood(ROOTS)
    grass['height'] = random.random()+ 0.1


plott = tools.PlotClass(world, rankIDLayer)
    
while True:
    tt = time.time()
    [grass.grow() for grass in world.iterNodes(GRASS)]
    world.papi.updateGhostNodes(propertyList=['height'])
    print(str(time.time() -tt) + ' s')
    


    plott.update(world)
