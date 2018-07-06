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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def plotGraph(world, agentTypeID, linkTypeID=None, attrLabel=None):
    linesToDraw = list()
    positions = world.getNodeAttr('pos', nodeTypeID=agentTypeID)
    
    #print(nAgents)
    plt.figure('graph')
    plt.clf()
    ax = plt.subplot(111)
    for agent in world.iterNodes(agentTypeID):
        pos = agent.attr['pos'][0]
        
        
        
        
        
        peerDataIDs   = np.asarray(agent.getPeerIDs(linkTypeID)) - world.maxNodes
        if len(peerDataIDs)> 0:
            peerPositions = positions[peerDataIDs]
            for peerPos in peerPositions:
                
                linesToDraw.append([[pos[0], pos[1]], [peerPos[0], peerPos[1]]])
    lc = mc.LineCollection(linesToDraw, colors='b', lw=.1) 
    if attrLabel is not None:
        values = world.getNodeAttr(attrLabel, nodeTypeID=agentTypeID)
        values = values / np.max(values)
        #print(values)
        plt.scatter(positions[:,0], positions[:,1], s = 15, c = values,zorder=2)
    else:
        plt.scatter(positions[:,0], positions[:,1], s = 15, zorder=2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)      
    plt.tight_layout()  
    #plt.colorbar()
    plt.draw()
