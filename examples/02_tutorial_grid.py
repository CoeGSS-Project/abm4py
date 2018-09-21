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
import matplotlib.pyplot as plt

from abm4py import World, Location, core

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
for i in range(3):
    ax = plt.subplot(1,3,1+i)
    connMask = world.grid.computeConnectionList(radius=1.5+i)
    world.grid.init(gridMask, connMask, Location)
    
    core.plotGraph(world, GRIDNODE, LINK, ax=ax)
    ax.set_title('Radius= ' + str(1.5+i))
