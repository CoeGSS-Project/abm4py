#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:57:17 2018

@author: andreas geiges

buzz diffuion model
"""

#%% load modules

import sys 
import os
import numpy as np
import logging as lg
import time

import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../../lib/')

import lib_gcfabm_prod as LIB #, GhostAgent, World,  h5py, MPI
import core_prod as core

#%% CONFIG
N_AGENTS   = 200
N_STEPS    = 1000
MAX_EXTEND = 50

IMITATION = 10
INNOVATION = .1

DEBUG = True

BLUE = plt.get_cmap('Blues')(.5)
RED  = plt.get_cmap('RdPu_r')(1)

#%% setup
simNo, outputPath = core.setupSimulationEnvironment()

world = LIB.World(simNo,
              outputPath,
              spatial=True,
              nSteps=N_STEPS,
              maxNodes=1e4,
              maxEdges=1e5,
              debug=DEBUG)

AGENT = world.registerNodeType('agent' , AgentClass=LIB.Agent,
                               staticProperties  = [('gID', np.int32,1),
                                                    ('pos', np.int16, 2)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
#%% Init of edge types
CON_AA = world.registerLinkType('ag-ag', AGENT,AGENT)

for iAgent in range(N_AGENTS):
    
    x,y = np.random.randint(0, MAX_EXTEND, 2)
    agent = LIB.Agent(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE)
    agent.register(world)
    
#%% Plotting
class PlotClass():
    
        
    def __init__(self, positions):
        
        plt.ion()
        self.fig = plt.figure(1)
        plt.clf()
    
    
    
        
        
        plt.subplot(1,2,1)
        self.h_plot = plt.plot([0,1],[1,1])[0]
        plt.xlim([0,1000])
        plt.ylim([0,1])
        plt.subplot(1,2,2)
        self.h_scatter = plt.scatter(positions[:,0], positions[:,1], s = 25, c = np.zeros(positions.shape[0]))
    


    def update(self, istep, plotDataList, scatterColorList):
        self.h_plot.set_data(range(istep),fracList)
        self.h_scatter.set_facecolor(scatterColorList)
        plt.draw()
        self.fig.canvas.flush_events()
    
    
positions = world.getNodeAttr('pos',nodeTypeID=AGENT)


#%% Scheduler
iStep = 0
fracList = list()
switched = world.getNodeAttr('switch',nodeTypeID=AGENT)

ploting = PlotClass(positions)
while True:
    tt =time.time()
    iStep+=1
    switched = world.getNodeAttr('switch',nodeTypeID=AGENT)
    switchFraction = np.sum(switched) / N_AGENTS
    fracList.append(switchFraction)
    
    if switchFraction == 1 or iStep == N_STEPS:
        break
    
    
    for agent, randNum in zip(world.iterNodes(AGENT), np.random.random(N_AGENTS)*1000):
        
        if agent.attr['switch'] == 0:
            if randNum < INNOVATION + ( IMITATION * (switchFraction )):
                agent.attr['switch'] = 1
                agent.attr['color'] = RED
            
    if iStep%10 == 0:
        ploting.update(iStep, fracList, world.getNodeAttr('color',nodeTypeID=AGENT))
    #time.sleep(.1)
    print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))