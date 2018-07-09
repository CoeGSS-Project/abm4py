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
sys.path.append('../')


from lib import World, Agent, Location #, GhostAgent, World,  h5py, MPI
from lib.enhancements import Neighborhood, Collective, Mobil
from lib import core

import tools_for_02 as tools

#%% SETUP
EXTEND = 100
N_SHEEPS = 70
N_PACKS  = 1
N_WOLFS  = 10
W_SHEEP = 5. # Reproduction weight for sheep
W_WOLF = 4.  # Reproduction weight for wolves
APPETITE = .30  # Percentage of grass height that sheep consume


#%% Sheep class
class Grass(Location, Neigbourhood, Collective):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Location.__init__(self, world, **kwAttr) #hier stand LIB.Agent, warum?
        Neigbourhood.__init__(self, world, **kwAttr)
        Collective.__init__(self, world, **kwAttr)
        
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
                    self.attr['height'] += 0.05
                    
                    if self.attr['height'] > 0.1:
                        break
                    
        self.attr['height'] = min(self.attr['height']*1.1, 1.)
#        if self.attr['height'] > 0.7:
#             [neigLoc.add(0.05) for neigLoc in self.groups['rooted'] if neigLoc.attr['height'] < 0.1]   
#        if self.attr['height'] > 0.7:
#            for neigLoc in self.groups['rooted']:
#                if neigLoc.attr['height'] < 0.1:
#                    neigLoc.add(0.05)
                    
       #if neigLoc in self.groups['rooted'] and neigLoc.attr['height'] < 0.1:
           

class Sheep(Agent, Mobil):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Agent.__init__(self, world, **kwAttr)
        Mobil.__init__(self, world, **kwAttr)

    def register(self,world):

        Agent.register(self, world)
        self.loc = locDict[(x,y)]
        world.addLink(LINK_SHEEP, self.nID, self.loc.nID)
        world.addLink(LINK_SHEEP, self.loc.nID, self.nID)
        
    def eat(self):
        """ Jette
        
        The function eat lets a sheep eat a percentage of the grass it is 
        standing on. This is subtracted from the grass's length and added
        to the sheep's weight.
        
        """
        food = self.loc.attr['height'] * APPETITE 
        self.loc.attr['height'] -= food
        self.attr['weight'] += food
        
    def move(self):
        """ Jette
        
        The function move lets the sheep move to a new position randomly 
        drawn around its current position. It is made sure, that they do 
        not leave the premises. Also links to old neighbours are deleted 
        and new links established. Additionally the sheep looses 0.1 units 
        of weight.
        
        """
        (dx,dy) = np.random.randint(-2,3,2)
        newX, newY = (self.attr['pos'] + [ dx, dy])[0] 
        #warum oben runde und hier eckige Klammern um dx, dy
        
        newX = min(max(0,newX), EXTEND-1)
        newY = min(max(0,newY), EXTEND-1)
        
#        self.attr['pos'] = [ newX, newY]
#        world.delLinks(LINK_SHEEP, self.nID, self.loc.nID)
#        world.delLinks(LINK_SHEEP, self.loc.nID, self.nID)
#        self.loc =  locDict[( newX, newY)]
#        world.addLink(LINK_SHEEP, self.nID, self.loc.nID)
#        world.addLink(LINK_SHEEP, self.loc.nID, self.nID)
        
        Mobil.move(self, newX, newY, LINK_SHEEP)
        self.attr['weight'] -= .1
        
    def step(self, world):
        
        self.eat()
        if random.random() > sqrt(self.loc.attr['height']):
            self.move()
        
        # If a sheep has enough weight and at least one other sheep close 
        # this produces a new sheep which is registered to the world
        # and the old sheep is back to initial weight.
        if self.attr['weight'] > W_SHEEP and len(self.loc.getPeerIDs(LINK_SHEEP))>0:
            newSheep = Sheep(world,
                             pos=self.attr['pos'],
                             weight=1.0)
            newSheep.register(world)
            
            self.attr['weight'] = 1.0
        
        # If a sheep starves to death it is deleted from the world.
        elif self.attr['weight'] < 0:
            self.delete(world)        
        
class Wolf(Agent):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['pos'])
        Agent.__init__(self, world, **kwAttr)


    def register(self,world):
        Agent.register(self, world)
        self.loc = locDict[(x,y)]
        world.addLink(LINK_WOLF, self.nID, self.loc.nID)
        world.addLink(LINK_WOLF, self.loc.nID, self.nID)
        
    def hunt(self):
        """ Jette
        
        The function hunt lets a wolf choose its prey randomly from a list
        of sheep around it. The sheep is pronounced dead and the wolf gains
        five units of weight.
        
        """
        sheepList = self.loc.getPeerIDs(liTypeID=LINK_SHEEP)
        if len(sheepList) > 0:
            sheep = self.loc.getMember(random.choice(sheepList))
            sheep.delete(world)
            self.attr['weight'] += 5.0
            print('sheep died')
    
    def move(self, center):
        """ Jette
        
        The function move lets the wolves move in the same way as sheep. 
        The wolf though only looses 0.02 units of weight.
        """
        pos = self.attr['pos'][0]
        delta =  pos - center
        
        bias = np.clip((.5*delta)**3 / np.sqrt(np.sum( delta**2)),a_min = -5, a_max=5).astype(np.int)
        (dx,dy) = np.random.randint(-3,4,2) - bias
        newX, newY = (pos + [ dx, dy])
        
        newX = min(max(0,newX), EXTEND-1)
        newY = min(max(0,newY), EXTEND-1)
        self.attr['pos'] = [ newX, newY]
        world.delLinks(LINK_WOLF, self.nID, self.loc.nID)
        world.delLinks(LINK_WOLF, self.loc.nID, self.nID)
        self.loc =  locDict[( newX, newY)]
        world.addLink(LINK_WOLF, self.nID, self.loc.nID)
        world.addLink(LINK_WOLF, self.loc.nID, self.nID)
        self.attr['weight'] -= .02

    def step(self, world, wolfPack):
        
        self.move(center)
            
        # If the wolves weight goes below 1 unit it starts hunting.
        if self.attr['weight'] < 1.0:
            self.hunt()
        
        # If a wolf has enough weight and at least one other wolf close 
        # this produces a new wolf which is registered to the world. 
        # The paternal wolf's weight goes down to 2.5 units.
        if self.attr['weight'] > W_WOLF and len(self.loc.getPeerIDs(LINK_WOLF))>0:
            newWolf = Wolf(world,
                           pos=self.attr['pos'],
                           weight=2.5)
            newWolf.register(world)
            wolfPack.join('wolfs',newWolf)
            self.attr['weight'] = 2.5
        
        # If a wolf starves to death it is deleted from the world.
        elif self.attr['weight'] < 0:
            self.delete(world)
            wolfPack.leave('wolfs',self)

class WolfPack(Agent, Collective):
    
    def __init__(self, world, **kwAttr):
        Agent.__init__(self, world, **kwAttr)
        Collective.__init__(self, world, **kwAttr)
        
        self.registerGroup('wolfs', [])
        
    def computeCenter(self):
        self.attr['center'] = np.mean(np.asarray([wolf.attr['pos'][0] for wolf in self.iterGroup('wolfs')]),axis=0)
        
        
        return self.attr['center'][0]
    
    
    def separate(self, world):
        
        newPack = WolfPack(world,
                           center=self.attr['center'][0])
        newPack.register(world)
        
        for wolf in self.iterGroup('wolfs'):
            if random.random() > .5:
                self.leave('wolfs', wolf)
                newPack.join('wolfs', wolf)
        newPack.computeCenter()
        
#%%
world = World(agentOutput=False,
                  maxNodes=100000,
                  maxLinks=200000)

world.setParameter('extend', EXTEND)
#%% register a new agent type with four attributes
GRASS = world.registerAgentType('grass' , AgentClass=Grass,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('height', np.float32, 1)])

SHEEP = world.registerAgentType('sheep' , AgentClass=Sheep,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('weight', np.float32, 1)])


WOLF = world.registerAgentType('wolf' , AgentClass=Wolf,
                               staticProperties  = [('pos', np.int16, 2)],
                               dynamicProperties = [('weight', np.float32, 1)])

WOLFPACK = world.registerAgentType('wolfPack' , AgentClass=WolfPack,
                               staticProperties  = [],
                               dynamicProperties = [('center', np.float32, 2),
                                                    ('nWolfs', np.int16, 1)])

#%% register a link type to connect agents
LINK_SHEEP = world.registerLinkType('grassLink', SHEEP, GRASS)
LINK_WOLF = world.registerLinkType('grassLink', WOLF, GRASS)

ROOTS = world.registerLinkType('roots',GRASS, GRASS, staticProperties=[('weig',np.float32,1)])
IDArray = np.zeros([EXTEND, EXTEND])

tt = time.time()
for x in range(EXTEND):
    for y in range(EXTEND):
        
        grass = Grass(world, 
                      pos=(x,y),
                      height=random.random())
        grass.register(world)
        world.registerLocation(grass, x,y)
        IDArray[x,y] = grass.nID
timePerAgent = (time.time() -tt ) / world.nAgents(GRASS)
print(timePerAgent)
connBluePrint = world.spatial.computeConnectionList(radius=1.5)
world.spatial.connectLocations(IDArray, connBluePrint, ROOTS, Grass)

for grass in world.iterNodes(GRASS):
    grass.reComputeNeighborhood(ROOTS)

del grass

# Jette: Sheep and wolves are assigned locations and registered to the world.

locDict = world.getLocationDict()
for iSheep in range(N_SHEEPS):
    (x,y) = np.random.randint(0,EXTEND,2)
    
    sheep = Sheep(world,
                  pos=(x,y),
                  weight=1.0)
    sheep.register(world)
    
del sheep
for iPack in range(N_PACKS):
    wolfPack = WolfPack(world,
                        center=(50,50))
    wolfPack.register(world)
    
    for iWolf in range(int(N_WOLFS/N_PACKS)):
        (x,y) = wolfPack.attr['center'][0] + np.random.randint(-1,2,2)
        
        wolf = Wolf(world,
                      pos=(x,y),
                      weight=1.0)
        wolf.register(world)
        wolfPack.join('wolfs',wolf)

wolfPack.computeCenter()        

del wolfPack, wolf 
plott = tools.PlotClass(world)
    
#%%
iStep = 0
while True:
    iStep +=1
    tt = time.time()
    # Every five steps the grass grows.
    if iStep%5 == 0:
        
        [grass.grow()for grass in world.iterNodes(GRASS)]
            
        
    
    [sheep.step(world) for sheep in world.iterNodes(agTypeID=SHEEP)]
       
        
    
    
    for wolfPack in world.iterNodes(agTypeID=WOLFPACK):
        center = wolfPack.computeCenter()   
        
        wolfPack.attr['nWolfs'] = len(wolfPack.groups['wolfs'])
        
        [wolf.step(world, wolfPack) for wolf in wolfPack.iterGroup('wolfs')]
            
        if wolfPack.attr['nWolfs'] > 20:
            wolfPack.separate(world)
        
        
        
    # This updates the plot.        
    pos = world.getAgentAttr('pos', agTypeID=SHEEP)
    if pos is not None:
        np.clip(pos, 0, EXTEND, out=pos)
        pos = world.setAgentAttr('pos', pos, agTypeID=SHEEP)
        plott.update(world)
    
    # This gives the number of sheep, the number of wolves and of these
    # the number of hunting wolves as strings in the console.
    nHunting = np.sum(world.getAgentAttr('weight', agTypeID=WOLF) <1.0)        
    print(str(time.time() - tt) + ' s')
    #print(str(world.nAgents(SHEEP)) + ' - ' + str(world.nAgents(WOLF)) + '(' + str(nHunting) + ')')
    iStep +=1