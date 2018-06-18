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
sys.path.append('../lib/')

import lib_gcfabm_prod as LIB #, GhostAgent, World,  h5py, MPI
import core_prod as core

#%% CONFIG
N_AGENTS   = 2000
N_STEPS    = 1000
MAX_EXTEND = 50

IMITATION = 10
INNOVATION = .1

DEBUG = True

BLUE = plt.get_cmap('Blues')(.5)
RED  = plt.get_cmap('RdPu_r')(1)

#%% setup
simNo, outputPath = core.setupSimulationEnvironment()

world = ...

AGENT = ...


#%% init of agents

for iAgent in range(N_AGENTS):
    
    ...
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
        self.h_plot.set_data(range(istep),plotDataList)
        self.h_scatter.set_facecolor(scatterColorList)
        plt.draw()
        self.fig.canvas.flush_events()
    
    



#%% Scheduler
iStep = 0
ploting = PlotClass(...)

while True:
    tt =time.time()
    iStep+=1
    # Stop condition
    
    ...
    
    # % Step dynamic
    
    ...
    

    # Plotting        
    if iStep%10 == 0:
        ploting.update(iStep, ..., ...)
    
    print('Step ' + str(iStep) +' finished after: ' + str(time.time()-tt))