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
import sys 
import numpy as np
import time
sys.path.append('../')

from abm4py import World, Location #, GhostAgent, World,  h5py, MPI
from abm4py import core



#%% Class definition
class Neighbor(Location):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Location.__init__(self, world, **kwAttr)
 
       
 #%% Setup
EXTEND   = 50
DO_PLOT = 0
       
world = World(agentOutput=False,
          maxNodes=1000000,
          maxLinks=1000000)

#%% register a new agent type with four attributes
nodeMap = np.zeros([EXTEND, EXTEND]) +1

NEIG = world.registerAgentType(AgentClass=Neighbor, agTypeStr='neigborhood',
                               staticProperties  = [('coord', np.int16, 2)],
                               dynamicProperties = [('property', np.float32, 1)])

LINK = world.registerLinkType('grassLink',NEIG, NEIG, dynamicProperties = [('weig', np.float32, 1)])

world.registerGrid(NEIG, LINK)   
connList      = world.grid.computeConnectionList(radius=10.5)
connBluePrint = world.grid.init(nodeMap, connList, Neighbor, LINK)

world.setAttrOfAgentType('property', 0., agTypeID=NEIG)

print('number of Neighbors: ' + str(world.countAgents(NEIG)))        
print('number of Links: ' + str(world.countLinks(LINK)))        
tt = time.time()
for neig in world.getAgentsByType(NEIG):
    for peer in neig.getGridPeers():
        peer.attr['property'] += 1.
print('Iteration over neighbors took ' + str(time.time() - tt) + ' s')


if DO_PLOT:        
    core.plotGraph(world, agentTypeID=NEIG, liTypeID=LINK,  attrLabel='property')        

    