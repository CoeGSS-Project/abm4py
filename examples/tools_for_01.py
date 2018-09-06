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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

def plotGraph(world, agentTypeID, liTypeID=None, attrLabel=None):
    linesToDraw = list()
    positions = world.getAttrOfAgentType('pos', agTypeID=agentTypeID)
    
    #print(nAgents)
    plt.ion()
    fig = plt.figure('graph')
    plt.clf()
    ax = plt.subplot(111)
    for agent in world.getAgentsByType(agentTypeID):
        pos = agent.attr['pos']
        peerDataIDs = np.asarray(agent.getPeerIDs(liTypeID)) - world.maxNodes
        if len(peerDataIDs) > 0:
            peerPositions = positions[peerDataIDs]
            for peerPos in peerPositions:
                linesToDraw.append([[pos[0], pos[1]], [peerPos[0], peerPos[1]]])

    lc = mc.LineCollection(linesToDraw, colors='b', lw=.1) 
    if attrLabel is not None:
        values = world.getAttrOfAgentType(attrLabel, agTypeID=agentTypeID)
        values = values / np.max(values)
        #print(values)
        plt.scatter(positions[:, 0], positions[:, 1], s = 15, c = values, zorder = 2)
    else:
        plt.scatter(positions[:, 0], positions[:, 1], s = 15, zorder = 2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)      
    plt.tight_layout()
    #plt.colorbar()
    plt.draw()
    fig.canvas.flush_events()
