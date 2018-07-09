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
    
        
    def __init__(self, world, rankIDLayer):
        
        plt.ion()
        self.fig = plt.figure('output')
        self.rankIDLayer = rankIDLayer
        self.extend = world.getParameter('extend')
        self.globalArray = np.zeros([self.extend, self.extend])
        plt.clf()
        
        
        self.gatherData(world)
        
        if world.isRoot:
            self.hh_area = plt.pcolormesh(self.globalArray, cmap='summer_r')
            
            plt.xlim(0, self.extend)
            plt.ylim(0, self.extend)
            plt.clim(0, np.max(self.globalArray))
            plt.colorbar(self.hh_area)

    def gatherData(self, world):
       
        parts = world.papi.comm.gather(world.getAgentAttr('height', agTypeID=1))
        if world.isRoot:
            for rank, part in enumerate(parts):
                #print(rank)
                #print(part)
                self.globalArray[self.rankIDLayer==rank] = part
    
    def update(self, world):
        self.gatherData(world)
        if world.isRoot:
            self.hh_area.set_array(self.globalArray.flatten())
            plt.clim(np.min(self.globalArray), np.max(self.globalArray))
            plt.draw()
            self.fig.canvas.flush_events()