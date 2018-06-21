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
import random

import matplotlib.pyplot as plt
home = os.path.expanduser("~")
sys.path.append('../../lib/')

import lib_gcfabm as LIB #, GhostAgent, World,  h5py, MPI
import core as core
import tools
#%% CONFIG
N_AGENTS   = 1000
N_STEPS    = 1000
MAX_EXTEND = 50

IMITATION = 15
INNOVATION = .2

N_FRIENDS  = 10

DEBUG = True
DO_PLOT  = True


BLUE = plt.get_cmap('Blues')(.3)
RED  = plt.get_cmap('RdPu_r')(.1)

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
                                                    ('pos', np.float32, 2),
                                                    ('imit', np.float16, 1),
                                                    ('inno', np.float16,1)],
                               dynamicProperties = [('switch', np.int16, 1),
                                                    ('color', np.float16,4)])
#%% Init of edge types
LI_AA = world.registerLinkType('ag-ag', AGENT,AGENT)

origins = [np.asarray(x) for x in [[25,25], [75,25], [25,75], [75,75]]]
individualDeltas  = np.clip(np.random.randn(2, N_AGENTS)*10, -25, 25)

for iAgent in range(N_AGENTS):
    
    sector = random.choice([0,1,2,3])
    
    origin = origins[sector]
    x,y = origin + individualDeltas[:,iAgent]
    inno = random.random() * INNOVATION
    imit = random.normalvariate(IMITATION,2 ) 
    agent = LIB.Agent(world,
                      pos=(x, y),
                      switch = 0,
                      color = BLUE,
                      imit = imit,
                      inno = inno)
    
    agent.register(world)


#%% creation of spatial proximity network
    
positions = world.getNodeAttr('pos', nodeTypeID=AGENT)
agIDList  = world.getNodeIDs(AGENT)
innovationVal = world.getNodeAttr('inno', nodeTypeID=AGENT).astype(np.float64)

for agent in world.iterNodes(AGENT):
    #opt1
    #weig = np.sum((positions - agent.attr['pos'])**2,axis=1)
    #opt2
    weig = np.abs((innovationVal - agent.attr['inno'])**20)
    weig = np.divide(1.,weig, out=np.zeros_like(weig), where=weig!=0)
    weig = weig / np.sum(weig)
    
    friendIDs = np.random.choice(agIDList, N_FRIENDS, replace=False, p=weig)
    
    [agent.addLink(ID, linkTypeID = LI_AA) for ID in friendIDs]
    

    
    
positions = world.getNodeAttr('pos',nodeTypeID=AGENT)
positions[:,0] = world.getNodeAttr('inno',nodeTypeID=AGENT)
positions[:,1] = world.getNodeAttr('imit',nodeTypeID=AGENT)
#%% Scheduler
iStep = 0
fracList = list()
fracPerSector = {1:[], 2:[], 3:[],4:[]}

switched = world.getNodeAttr('switch',nodeTypeID=AGENT)



if DO_PLOT:
    plotting = tools.PlotClass(positions, world, AGENT, LI_AA)
while True:
    tt =time.time()
    iStep+=1
    switched = world.getNodeAttr('switch',nodeTypeID=AGENT)
    switchFraction = np.sum(switched) / N_AGENTS
    fracList.append(switchFraction)
    
    
    if switchFraction == 1 or iStep == N_STEPS:
        break
    tools.printfractionExceed(switchFraction, iStep)
    
    nodesToIter = world.filterNodes(AGENT, 'switch', 'eq', 0)
    randValues  = np.random.random(len(nodesToIter))*1000
    for agent, randNum in zip(world.iterNodes(localIDs=nodesToIter),randValues) :
        
        if agent.attr['switch'] == 0:
            
            switchFraction = np.sum(agent.getPeerAttr('switch',LI_AA)) / N_FRIENDS
            inno, imit = agent.attr[['inno','imit']][0]
            
            if randNum < inno + ( imit * ( switchFraction)):
                agent.attr['switch'] = 1
                agent.attr['color'] = RED
                plotting.add(iStep,inno)
            
    if DO_PLOT and iStep%10 == 0:
        plotting.update(iStep, fracList, world.getNodeAttr('color',nodeTypeID=AGENT))
    #time.sleep(.1)
    #print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))