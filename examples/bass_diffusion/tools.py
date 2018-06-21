#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:33:12 2018

@author: gcf
"""
import sys 
import os
import numpy as np
import logging as lg
import time
import random

import matplotlib.pyplot as plt


over50 = False
over80 = False
over90 = False

#%% Plotting
class PlotClass():
    
        
    def __init__(self, positions, world, AGENT, LI_AA):
        
        from matplotlib import collections  as mc

        tt = time.time()
        plt.ion()
        self.fig = plt.figure(1)
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
        linesToDraw = list()
        plt.cla()
        for agent in world.iterNodes(AGENT):
            pos = agent.attr['pos'][0]
            pos = positions[agent.dataID]
            
            peerDataIDs     = np.asarray(agent.getPeerIDs(LI_AA)) - world.maxNodes
            peerPositions = positions[peerDataIDs]
            for peerPos in peerPositions:
                
                linesToDraw.append([[pos[0], pos[1]], [peerPos[0], peerPos[1]]])
        lc = mc.LineCollection(linesToDraw, colors='b', lw=.1) 

        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        
        
        self.h_scatter = plt.scatter(positions[:,0], positions[:,1], s = 25, c = np.zeros(positions.shape[0]),zorder=2)
        plt.tight_layout()
        print ('Init plot done in ' + str(time.time()-tt) + ' s')


    def update(self, istep, plotDataList, scatterColorList):
        self.h_plot.set_data(range(istep),plotDataList)
        self.h_scatter.set_facecolor(scatterColorList)
        self.ax2.scatter(self.xPos, self.yPos)
        self.xPos = list()
        self.yPos = list()
        plt.draw()
        self.fig.canvas.flush_events()
    
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