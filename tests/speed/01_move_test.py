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
import random
sys.path.append('../../')

from lib import World, Location, Agent #, GhostAgent, World,  h5py, MPI
from lib.traits import Mobile
from lib import core



#%% Class definition
class Walker(Agent, Mobile):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Agent.__init__(self, world, **kwAttr)
        Mobile.__init__(self, world, **kwAttr)

    def register(self,world):
        Agent.register(self, world)
        self.loc = world.getLocationDict()[tuple(self.get('pos'))]
        #world.addLink(ANCHOR, self.nID, self.loc.nID)
        world.addLink(ANCHOR, self.loc.nID, self.nID)
        
    def randomWalk(self):
        (dx,dy) = np.random.randint(-2,3,2)
        newX, newY = (self.attr['pos'] + [ dx, dy])[0]
        
        newX = min(max(0,newX), EXTEND-1)
        newY = min(max(0,newY), EXTEND-1)
        
#        self.attr['pos'] = [ newX, newY]
#        world.delLinks(LINK_SHEEP, self.nID, self.loc.nID)
#        world.delLinks(LINK_SHEEP, self.loc.nID, self.nID)
#        self.loc =  locDict[( newX, newY)]
#        world.addLink(LINK_SHEEP, self.nID, self.loc.nID)
#        world.addLink(LINK_SHEEP, self.loc.nID, self.nID)
        
        Mobile.move(self, newX, newY, ANCHOR)
       
 #%% Setup
EXTEND   = 50
DO_PLOT = False
N_WALKERS = 5000
N_STEPS   = 100
       
world = World(agentOutput=False,
          maxNodes=100000,
          maxLinks=500000)

#%% register a new agent type with four attributes
nodeMap = np.zeros([EXTEND, EXTEND])

LOC = world.registerAgentType('location' , AgentClass=Location,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('property', np.float32, 1)])

LINK = world.registerLinkType('link',LOC, LOC, dynamicProperties = [('weig', np.float32, 1)])


WKR = world.registerAgentType('walker' , AgentClass=Walker,
                               dynamicProperties =  [('pos', np.int16, 2)])

ANCHOR  = world.registerLinkType('link',LOC, WKR)

tt = time.time()

connList      = world.spatial.computeConnectionList(radius=1.5)
connBluePrint = world.spatial.initSpatialLayer(nodeMap, connList, Location, LINK)

world.setAttrOfAgentType('property', 0., agTypeID=LOC)
print('Spatial layer created in ' + str(time.time() -tt) )   
print('Number of Locations: ' + str(world.nAgents(LOC)))        
print('Number of spatial links: ' + str(world.nLinks(LINK)))        

locList = world.getAgents.byType(LOC)
tt = time.time()
for iWalker in range(N_WALKERS):

    loc = random.choice(locList)    
    walker = Walker(world,
                  pos=tuple(loc.attr['pos'][0]))
    walker.loc = loc
    walker.register(world)
print('Walkers created in ' + str(time.time() -tt) )   
print('Number of Walkers: ' + str(world.nAgents(WKR)))        
print('Number of locating links: ' + str(world.nLinks(ANCHOR)))     

timeReq=list()

for iSteps in range(N_STEPS):  
    tt = time.time()
    [walker.randomWalk() for walker in world.getAgents.byType(WKR)] 
    timeReq.append(time.time() -tt)
    if DO_PLOT:        
        core.plotGraph(world, agentTypeID=WKR)
if not DO_PLOT:  
    print('Average: {:3.4f} s STD: {:3.4f} s'.format(np.mean(timeReq[1:]), np.std(timeReq[1:])))

