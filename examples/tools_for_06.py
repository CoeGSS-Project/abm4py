#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:36:39 2018

@author: gcf
"""

import sys 
import os
import numpy as np
import logging as lg
import time
import random

import matplotlib.pyplot as plt
from matplotlib import collections  as mc


class PlotClass():
    
        
    def __init__(self, world):
        
        plt.ion()
        self.fig = plt.figure('spatial')
        plt.clf()
        plt.subplot(1,2,1)
        extend = world.getParameters()['extend']
        
        
        pos = world.getAttrOfFilteredAgentType('pos', lambda a: a['sick'], 2)
        if len(pos) > 0:
            self.h_sicks = plt.scatter(pos[:,1],pos[:,0], c='r', s = 35, marker='s',zorder=2)
        
        pos = world.getAttrOfFilteredAgentType('pos', lambda a: a['sick'] == False, 2)
        if len(pos) > 0:
            self.h_healths = plt.scatter(pos[:,1],pos[:,0], c='g', s = 35, marker='s',zorder=2)
        
        pos = world.getAttrOfFilteredAgentType('pos', lambda a: a['remainingImmunity'], 2)
        if len(pos) > 0:
            self.h_immunes = plt.scatter(pos[:,1],pos[:,0], c='k', s = 35, marker='s',zorder=2)
        
        

        plt.xlim(0, extend)
        plt.ylim(0, extend)
        
    
        plt.subplot(1,2,2)
        
        self.sheepmax = 0
        from collections import deque
        self.sicks = deque([0]*100)
        self.healths  = deque([0]*100)
        self.immunes  = deque([0]*100)
        
        self.timeSicks = plt.plot(self.sicks, c='r')
        self.timeHealths = plt.plot(self.healths, c='g')
        self.timeImmunes = plt.plot(self.immunes, c='k')
        plt.ylim([0, 300])
        plt.legend(['number of sick people', 'number of healthy people', 'number of immune people'])
        
    def update(self, world):
        pos = world.getAttrOfFilteredAgentType('pos', lambda a: a['sick'], 2)
        if len(pos) > 0:
            self.h_sicks.set_offsets(np.c_[pos[:,1],pos[:,0]])

        pos = world.getAttrOfFilteredAgentType('pos', lambda a: a['sick'] == False, 2)
        if len(pos) > 0:
            self.h_healths.set_offsets(np.c_[pos[:,1],pos[:,0]])

        pos = world.getAttrOfFilteredAgentType('pos', lambda a: a['remainingImmunity'], 2)
        if len(pos) > 0:
            self.h_immunes.set_offsets(np.c_[pos[:,1],pos[:,0]])
        
        plt.draw()
        
        self.sicks.popleft()
        self.sicks.append(world.countAgents(lambda a: a['sick'], 2))
        
        self.healths.popleft()
        self.healths.append(world.countAgents(lambda a: a['sick'] == False, 2))
        
        self.immunes.popleft()
        self.immunes.append(world.countAgents(lambda a: a['remainingImmunity'] == False, 2))
                    
        self.timeSicks[0].set_ydata(self.sicks)
        self.timeHealths[0].set_ydata(self.healths)
        self.timeImmunes[0].set_ydata(self.immunes)
        
        self.fig.canvas.flush_events()
        #self.fig2.canvas.flush_events()
        plt.draw()
