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
        self.fig = plt.figure('output')
        plt.clf()
        extend = world.getParameter('extend')
        
        grass = np.reshape(world.getNodeAttr('height', nodeTypeID=1),[extend, extend])
        
        
        self.hh_area = plt.pcolormesh(grass, cmap='summer_r',zorder=-1)
        
        pos = world.getNodeAttr('pos', nodeTypeID = 2)
        #print(pos.shape)
        self.hh_sheeps = plt.scatter(pos[:,1],pos[:,0], c='w', s = 35, marker='s',zorder=2)
        self.hh_wolfs  = plt.scatter(pos[:,1],pos[:,0], c='k', s = 35, marker='s',zorder=2)
        plt.xlim(0, extend)
        plt.ylim(0, extend)
        plt.clim(0,1)
        plt.colorbar(self.hh_area)
    
    def update(self, world):

        pos = world.getNodeAttr('pos', nodeTypeID = 2)
        
        self.hh_sheeps.set_offsets(np.c_[pos[:,1],pos[:,0]])
        pos = world.getNodeAttr('pos', nodeTypeID = 3)
        self.hh_wolfs.set_offsets(np.c_[pos[:,1],pos[:,0]])
        self.hh_area.set_array(world.getNodeAttr('height', nodeTypeID=1))
        plt.draw()
        self.fig.canvas.flush_events()