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
import numpy as np
import time
import random
from math import sqrt

from abm4py import World, Agent, Location #, GhostAgent, World,  h5py, MPI
from abm4py.traits import  Mobile
from abm4py.future_traits import Collective

import tools_for_03 as tools


#%% SETUP
EXTEND = 100
N_SHEEPS = 300
N_PACKS  = 1
N_WOLFS  = 10
W_SHEEP = 10. # Reproduction weight for sheep
W_WOLF = 4.  # Reproduction weight for wolves
APPETITE = .30  # Percentage of grass height that sheep consume

DO_PLOT = True

#%% Sheep class
class Grass(Location , Collective):

    def __init__(self, world, **kwAttr):
        #print(kwAttr['coord'])
        Location.__init__(self, world, **kwAttr) #hier stand LIB.Agent, warum?
        Collective.__init__(self, world, **kwAttr)

    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent._graph an can be shared/viewed by other agents and acessed via 
        the global scope of the world class.
        All static and dynamic attributes can be accessed by the agent by:
            1) agent.get('attrLabel') / agent.set('attrLabel', value)
            2) agent.attr['attrLabel']
            3) agent.attr['attrLabel']
        """
        classDesc = dict()
        classDesc['nameStr'] = 'grass'
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
        """        
        The function grow lets the grass grow by ten percent.
        If the grass height is smaller than 0.1, and a neighboring patch has 
        grass higher than 0.7, the grass grows by .05. Then it grows by 
        10 percent.
        """
                
        if self.attr['height'] < 0.1:
            for neigLoc in self.getGridPeers():
                if neigLoc.attr['height'] > 0.9:
                    self.attr['height'] += 0.05
                    
                    if self.attr['height'] > 0.1:
                        break
                    
        self.attr['height'] = min(self.attr['height']*1.1, 1.)
           

class Sheep(Agent, Mobile):


    def __init__(self, world, **kwAttr):
        #print(kwAttr['coord'])
        Agent.__init__(self, world, **kwAttr)
        Mobile.__init__(self, world, **kwAttr)
        
        self.loc = locDict[(x,y)]
        #world.addLink(LINK_SHEEP, self.nID, self.loc.nID)
        world.addLink(LINK_SHEEP, self.loc.nID, self.nID)
    
    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent._graph an can be shared/viewed by other agents and acessed via 
        the global scope of the world class.
        All static and dynamic attributes can be accessed by the agent by:
            1) agent.get('attrLabel') / agent.set('attrLabel', value)
            2) agent.attr['attrLabel']
            3) agent.attr['attrLabel']
        """
        classDesc = dict()
        classDesc['nameStr'] = 'sheep'
        # Static properites can be re-assigned during runtime, but the automatic
        # IO is only logging the initial state
        classDesc['staticProperties'] =  []          
        # Dynamic properites can be re-assigned during runtime and are logged 
        # per defined time step intervall (see core.IO)
        classDesc['dynamicProperties'] = [('coord', np.int16, 2),
                                          ('weight', np.float32, 1)]     
        return classDesc

    
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
        newX, newY = self.attr['coord']+ [ dx, dy]
        #warum oben runde und hier eckige Klammern um dx, dy
        
        newX = min(max(0,newX), EXTEND-1)
        newY = min(max(0,newY), EXTEND-1)
        
#        self.attr['coord'] = [ newX, newY]
#        world.delLinks(LINK_SHEEP, self.nID, self.loc.nID)
#        world.delLinks(LINK_SHEEP, self.loc.nID, self.nID)
#        self.loc =  locDict[( newX, newY)]
#        world.addLink(LINK_SHEEP, self.nID, self.loc.nID)
#        world.addLink(LINK_SHEEP, self.loc.nID, self.nID)
        
        Mobile.move(self, newX, newY, LINK_SHEEP)
        self.attr['weight'] -= .1
        
    def step(self, world):
        
        self.eat()
        if random.random() > sqrt(self.loc.attr['height']):
            self.move()
        
        # If a sheep has enough weight and at least one other sheep close 
        # this produces a new sheep which is registered to the world
        # and the old sheep is back to initial weight.
        if self.attr['weight'] > W_SHEEP and len(self.loc.getPeerIDs(LINK_SHEEP))>0:
            world.registerAgent(Sheep(world,
                                      coord=self.attr['coord'],
                                      weight=1.0))
            self.attr['weight'] = 1.0
        
        # If a sheep starves to death it is deleted from the world.
        elif self.attr['weight'] < 0:
            self.delete(world)        
        
class Wolf(Agent, Mobile):

    def __init__(self, world, **kwAttr):
        Agent.__init__(self, world, **kwAttr)
        Mobile.__init__(self, world, **kwAttr)
        world.registerAgent(self)
        self.loc = locDict[(x,y)]
        world.addLink(LINK_WOLF, self.loc.nID, self.nID)

    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent._graph an can be shared/viewed by other agents and acessed via 
        the global scope of the world class.
        All static and dynamic attributes can be accessed by the agent by:
            1) agent.get('attrLabel') / agent.set('attrLabel', value)
            2) agent.attr['attrLabel']
            3) agent.attr['attrLabel']
        """
        classDesc = dict()
        classDesc['nameStr'] = 'wolf'
        # Static properites can be re-assigned during runtime, but the automatic
        # IO is only logging the initial state
        classDesc['staticProperties'] =  [('coord', np.int16, 2)]          
        # Dynamic properites can be re-assigned during runtime and are logged 
        # per defined time step intervall (see core.IO)
        classDesc['dynamicProperties'] = [('weight', np.float32, 1)]     
        return classDesc
    
    def hunt(self):
        """ 
        
        The function hunt lets a wolf choose its prey randomly from a list
        of sheep around it. The sheep is pronounced dead and the wolf gains
        five units of weight.
        
        """
        sheepList = self.loc.getPeerIDs(liTypeID=LINK_SHEEP)
        if len(sheepList) > 0:
            sheep = self.loc.getMember(random.choice(sheepList))
            sheep.delete(world)
            self.attr['weight'] += 5.0
            #print('sheep died')
    def move(self, center, hunt=False):
        """ 
        
        The function move lets the wolves move in the same way as sheep. 
        The wolf though only looses 0.02 units of weight.
        """
        pos = self['coord']
        delta =  pos - center
        
        bias = np.clip((.5*delta)**3 / np.sqrt(np.sum( delta**2)),a_min = -5, a_max=5).astype(np.int)
        if hunt:
            newX, newY = pos + np.random.randint(-5,6,2) - bias
        else:
            newX, newY = pos + np.random.randint(-3,4,2) - bias
        
        newX = min(max(0,newX), EXTEND-1)
        newY = min(max(0,newY), EXTEND-1)
        Mobile.move(self, newX, newY, LINK_WOLF)
        self.attr['weight'] -= .02

    def step(self, world, wolfPack):
        
        self.move(wolfPack.attr['center'])
            
        # If the wolves weight goes below 1 unit it starts hunting.
        if self.attr['weight'] < 1.5:
            self.hunt()
        
        # If a wolf has enough weight and at least one other wolf close 
        # this produces a new wolf which is registered to the world. 
        # The paternal wolf's weight goes down to 2.5 units.
        if self.attr['weight'] > W_WOLF and len(self.loc.getPeerIDs(LINK_WOLF))>0:
            newWolf = Wolf(world,
                           coord=self.attr['coord'],
                           weight=2.5)
            world.registerAgent(newWolf)
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

    def __descriptor__():
        """
        This desriptor defines the agent attributes that are saved in the 
        agent._graph an can be shared/viewed by other agents and acessed via 
        the global scope of the world class.
        All static and dynamic attributes can be accessed by the agent by:
            1) agent.get('attrLabel') / agent.set('attrLabel', value)
            2) agent.attr['attrLabel']
            3) agent.attr['attrLabel']
        """
        classDesc = dict()
        classDesc['nameStr'] = 'wolf'
        # Static properites can be re-assigned during runtime, but the automatic
        # IO is only logging the initial state
        classDesc['staticProperties'] =  []          
        # Dynamic properites can be re-assigned during runtime and are logged 
        # per defined time step intervall (see core.IO)
        classDesc['dynamicProperties'] = [('center', np.float32, 2),
                                          ('nWolfs', np.int16, 1)]   
        return classDesc

        
    def computeCenter(self):
        self.attr['nWolfs'] = len(self.groups['wolfs'])
        self.attr['center'] = np.mean(np.asarray([wolf.attr['coord'] for wolf in self.iterGroup('wolfs')]),axis=0)
        
        
        return self.attr['center']
    
    
    def separate(self, world):
        
        newPack = WolfPack(world,
                           center=self.attr['center'])
        world.registerAgent(newPack)
        
        for wolf in self.iterGroup('wolfs'):
            if random.random() > .5:
                self.leave('wolfs', wolf)
                newPack.join('wolfs', wolf)
        newPack.computeCenter()
        
    def step(self):
        self.computeCenter()   
        
        
        
        [wolf.step(world, self) for wolf in self.iterGroup('wolfs')]
            
        if self.attr['nWolfs'] > 20:
            self.separate(world)
            
            
#%% Register of the world class
world = World(agentOutput=False,
                  maxNodes=10000,
                  maxLinks=200000)

world.setParameter('extend', EXTEND)


#%% register a new agent type with four attributes
GRASS = world.registerAgentType(Grass)

SHEEP = world.registerAgentType(Sheep)

WOLF = world.registerAgentType(Wolf)

WOLFPACK = world.registerAgentType(WolfPack)


#%% register a link type to connect agents
ROOTS = world.registerLinkType('roots',GRASS, GRASS, staticProperties=[('weig',np.float32,1)])

LINK_SHEEP = world.registerLinkType('grassLink', SHEEP, GRASS)

LINK_WOLF = world.registerLinkType('grassLink', WOLF, GRASS)


#%% adding the grid to world
world.registerGrid(GRASS, ROOTS)

IDArray = np.zeros([EXTEND, EXTEND])

tt = time.time()
for x in range(EXTEND):
    for y in range(EXTEND):
        
        grass = Grass(world, 
                      coord=(x,y),
                      height=random.random())
        grass.register(world)
        
        IDArray[x,y] = grass.nID
timePerAgent = (time.time() -tt ) / world.countAgents(GRASS)
print(timePerAgent)

connBluePrint = world.grid.computeConnectionList(radius=1.5)
world.grid.connectNodes(IDArray, connBluePrint, ROOTS, Grass)


# Jette: Sheep and wolves are assigned locations and registered to the world.

locDict = world.grid.getNodeDict()
for iSheep in range(N_SHEEPS):
    (x,y) = np.random.randint(0,EXTEND,2)
    
    world.registerAgent(Sheep(world,
                              coord=(x,y),
                              weight=1.0))
    
for iPack in range(N_PACKS):
    wolfPack = WolfPack(world,
                        center=(50,50))
    world.registerAgent(wolfPack)
    
    for iWolf in range(int(N_WOLFS/N_PACKS)):
        (x,y) = wolfPack.attr['center'][0] + np.random.randint(-1,2,2)
        
        wolf = Wolf(world,
                      coord=(x,y),
                      weight=1.0)
        world.registerAgent(wolf)
        wolfPack.join('wolfs',wolf)

wolfPack.computeCenter()        

del wolfPack, wolf 

if DO_PLOT: 
    plott = tools.PlotClass(world)
    
#%%
import tqdm
iStep = 0
pbar = tqdm.tqdm()
while True:
    iStep +=1
    pbar.update(1)
    tt = time.time()
    # Every five steps the grass grows.
    if iStep%5 == 0:
        
        [grass.grow() for grass in world.getAgentsByType(GRASS)]
            
        
    
    [sheep.step(world) for sheep in world.getAgentsByType(SHEEP)]
       
        
    
    
    [wolfPack.step()  for wolfPack in world.getAgentsByType(WOLFPACK)]

        
        
        
        
    # This updates the plot.        
    if DO_PLOT: 
        pos = world.getAttrOfAgentType('coord', agTypeID=SHEEP)
        if pos is not None:
            np.clip(pos, 0, EXTEND, out=pos)
            pos = world.setAttrOfAgentType('coord', pos, agTypeID=SHEEP)
            plott.update(world)
    
    # This gives the number of sheep, the number of wolves and of these
    # the number of hunting wolves as strings in the console.
    nHunting = np.sum(world.getAttrOfAgentType('weight', agTypeID=WOLF) <1.0) 
    #grassHeight = np.sum(world.getAttrOfAgentType('height', agTypeID=GRASS))       
    #print(str(time.time() - tt) + ' s')
    print(str(world.countAgents(SHEEP)) + ' - ' + str(world.countAgents(WOLF)) + '(' + str(nHunting) + ')')
    iStep +=1
