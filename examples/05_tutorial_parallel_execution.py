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

import numpy as np
import time
import random

from gcfabm import World, Location, GhostLocation #, GhostAgent, World,  h5py, MPI
from gcfabm.traits import Parallel
from gcfabm.future_traits import Collective
from gcfabm import core

import tools_for_05 as tools

#%% SETUP
EXTEND = 20
RADIUS = 1.5

#%% Class definition
class Grass(Location, Collective, Parallel):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Location.__init__(self, world, **kwAttr)
        Collective.__init__(self, world, **kwAttr)
        Parallel.__init__(self, world, **kwAttr)

    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent graph an can be shared/viewed by other agents and acessed via 
        the global scope of the world class.
        All static and dynamic attributes can be accessed by the agent by:
            1) agent.get('attrLabel') / agent.set('attrLabel', value)
            2) agent.attr['attrLabel']
            3) agent.attr['attrLabel']
        """
        classDesc = dict()
        classDesc['nameStr'] = 'Grass'
        # Static properites can be re-assigned during runtime, but the automatic
        # IO is only logging the initial state
        classDesc['staticProperties'] =  [('coord', np.int16, 2)]          
        # Dynamic properites can be re-assigned during runtime and are logged 
        # per defined time step intervall (see core.IO)
        classDesc['dynamicProperties'] = [('height', np.float32, 1)]     
        return classDesc
 
    def add(self, value):
        
        self.attr['height'] += value


    def grow(self):
        currHeight = self.attr['height']
        for neigLoc in self.getGridPeers():
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
print(world.isParallel)
print(core.mpiRank)
print(core.mpiSize)
rankIDLayer = np.zeros([EXTEND, EXTEND]).astype(int)
if world.isParallel:
    print('parallel mode')
    print(core.mpiRank)
    if core.mpiSize == 4:
    
        rankIDLayer[EXTEND//2:,:EXTEND//2] = 1
        rankIDLayer[:EXTEND//2,EXTEND//2:] = 2
        rankIDLayer[:EXTEND//2,:EXTEND//2:] = 3

    elif core.mpiSize == 2:
        rankIDLayer[EXTEND//2:,:] = 1
    print(rankIDLayer)
else:
    print('non-parallel mode')
    
world.setParameter('extend', EXTEND)
GRASS = world.registerAgentType(AgentClass=Grass, GhostAgentClass=GhostGrass)
                                
ROOTS = world.registerLinkType('roots',GRASS, GRASS, staticProperties=[('weig',np.float32,1)])

world.registerGrid(GRASS, ROOTS)
connBluePrint = world.grid.computeConnectionList(radius=RADIUS)
world.grid.init((rankIDLayer*0)+1, connBluePrint, Grass, rankIDLayer)

for grass in world.getAgents.byType(GRASS):

    if np.all(grass['coord'] < 8):
        grass['height'] = random.random()+ 13.1    
    else:
        grass['height'] = random.random()+ 0.1


plott = tools.PlotClass(world, rankIDLayer)
    
while True:
    tt = time.time()
    [grass.grow() for grass in world.getAgents.byType(GRASS)]
    world.papi.updateGhostAgents(propertyList=['height'])
    print(str(time.time() -tt) + ' s')
    


    plott.update(world)
