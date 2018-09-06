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
#%%
import sys 
import os
import numpy as np
import logging as lg
import time
import random

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

over50 = False
over80 = False
over90 = False

#%% Plotting
class PlotClass():
    
        
    def __init__(self, positions, world, AGENT, LI_AA=None):

        tt = time.time()
        plt.ion()
        self.fig = plt.figure('output')
        #plt.clf()
    
        self.xPos = list()
        self.yPos = list()
    
        
        
        plt.subplot(2,2,1)
        self.h_plot = plt.plot([0,1],[1,1])[0]
        plt.xlim([0,1000])
        plt.ylim([0,1])
        self.ax2 = plt.subplot(2,2,3)
        #self.ax2.plot([0,0],'x')
        plt.cla()
        ax = plt.subplot(1,2,2)
        
        if LI_AA is not None:
            linesToDraw = list()
            plt.cla()
            for agent in world.getAgentsByType(AGENT):
                pos = agent.attr['pos'][0]
                pos = positions[agent.dataID]
                
                peerDataIDs     = np.asarray(agent.getPeerIDs(LI_AA)) - (world.maxNodes * AGENT)
                peerPositions   = positions[peerDataIDs]
                for peerPos in peerPositions:
                    
                    linesToDraw.append([[pos[0], pos[1]], [peerPos[0], peerPos[1]]])
            lc = mc.LineCollection(linesToDraw, colors='b', lw=.1) 
    
            ax.add_collection(lc)
            ax.autoscale()
            ax.margins(0.1)
        
        
        self.h_scatter = plt.scatter(positions[:,0], positions[:,1], s = 15, c = np.zeros(positions.shape[0]),zorder=2)
        plt.tight_layout()
        print ('Init plot done in ' + str(time.time()-tt) + ' s')

    def add_data(self, x, y):
        plt.subplot(2,2,1)
        plt.plot(x,y, 'o')

    def update(self, istep, plotDataList, scatterColorList):
        self.h_plot.set_data(range(istep),plotDataList)
        self.h_scatter.set_facecolor(scatterColorList)
        self.ax2.scatter(self.xPos, self.yPos, s = 15)
        self.xPos = list()
        self.yPos = list()
        plt.draw()
        self.fig.canvas.flush_events()
        #self.fig.canvas.draw()
    def add(self, xPos,yPos):
        self.xPos.append(xPos)
        self.yPos.append(yPos)
        
def printfractionExceed(switchFraction, iStep):
    global over50, over80, over90
    if not over50 and switchFraction > .5:
        print('50% reached in step ' + str(iStep))
        over50 = True
    if not over80 and switchFraction > .8:
        print('80% reached in step ' + str(iStep))
        over80 = True
    if not over90 and switchFraction > .9:
        print('90% reached in step ' + str(iStep))
        over90 = True        
        
        
def plotGraph(world, agentTypeID, liTypeID=None, attrLabel=None):
    linesToDraw = list()
    positions = world.getAttrOfAgentType('pos', agTypeID=agentTypeID)
    
    #print(nAgents)
    plt.figure('graph')
    plt.clf()
    ax = plt.subplot(111)
    for agent in world.getAgentsByType(agentTypeID):
        pos = agent.attr['pos'][0]
        
        
        
        
        
        peerDataIDs   = np.asarray(agent.getPeerIDs(liTypeID)) - world.maxNodes
        if len(peerDataIDs)> 0:
            peerPositions = positions[peerDataIDs]
            for peerPos in peerPositions:
                
                linesToDraw.append([[pos[0], pos[1]], [peerPos[0], peerPos[1]]])
    lc = mc.LineCollection(linesToDraw, colors='b', lw=.1) 
    if attrLabel is not None:
        values = world.getAttrOfAgentType(attrLabel, agTypeID=agentTypeID)
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
