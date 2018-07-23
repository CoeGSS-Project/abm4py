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
import numpy as np
import time
import random
import matplotlib.pyplot as plt

from gcfabm import World, Location, core

#%% SETUP
MAX_AGENTS = 1000
#%%
world = World(agentOutput=False,
                  maxNodes=MAX_AGENTS,
                  maxLinks=20000)

GRIDNODE = world.registerAgentType(Location)

LINK = world.registerLinkType('ink',GRIDNODE, GRIDNODE)

world.registerGrid(GRIDNODE, LINK)
gridMask = np.random.randint(0,2,[10,10])


plt.ion()
fig= plt.figure('graph')
plt.clf()
for i in range(4):
    ax = plt.subplot(2,2,1+i)
    connMask = world.grid.computeConnectionList(radius=1.5+i)
    world.grid.init(gridMask, connMask, Location)
    
    core.plotGraph(world, GRIDNODE, LINK, ax=ax)
    ax.set_title('Radius= ' + str(1.5+i))
