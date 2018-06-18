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

import lib_gcfabm as LIB #, GhostAgent, World,  h5py, MPI
import core as core

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
                                                    ('pos', np.float32, 2)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
#%% Init of edge types
LI_AA = world.registerLinkType('ag-ag', AGENT,AGENT)

for iAgent in range(N_AGENTS):
    
    x,y = np.random.randint(0, MAX_EXTEND, 2)
    agent = LIB.Agent(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE)
    agent.register(world)


#%% creation of spatial proximity network
    
positions = world.getNodeAttr('pos', nodeTypeID=AGENT)
agIDList  = world.getNodeIDs(AGENT)
nFriends  = 10

for agent in world.iterNodes(AGENT):
    weig = np.sum((positions - agent.attr['pos'])**2,axis=1)
    weig = np.divide(1.,weig, out=np.zeros_like(weig), where=weig!=0)
    weig = weig / np.sum(weig)
    
    friendIDs = np.random.choice(agIDList, nFriends, replace=False, p=weig)
    
    [agent.addLink(ID, linkTypeID = LI_AA) for ID in friendIDs]
    
#%% Plotting
class PlotClass():
    
        
    def __init__(self, positions, world):
        
        plt.ion()
        self.fig = plt.figure(1)
        plt.clf()
    
    
    
        
        
        plt.subplot(1,2,1)
        self.h_plot = plt.plot([0,1],[1,1])[0]
        plt.xlim([0,1000])
        plt.ylim([0,1])
        plt.subplot(1,2,2)
        for agent in world.iterNodes(AGENT):
            pos = agent.attr['pos'][0]
            
            for peerPos in agent.getPeerAttr('pos',linkTypeID=LI_AA):
                #print(str(peerPos))
                plt.plot([pos[0], peerPos[0]], [pos[1], peerPos[1]], linewidth=.2)
        
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

ploting = PlotClass(positions, world)
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